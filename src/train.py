import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import TobaccoDataset, get_model, get_transforms

def find_dataset_root(base_path):
    """
    Scans the directory for the class folders of Tobacco3482.
    Handles potential nested folders during copy.
    """
    target_classes = {'adve', 'email', 'form', 'letter', 'memo', 'news', 'note', 'report', 'resume', 'scientific'}
    
    queue = [base_path]
    while queue:
        curr = queue.pop(0)
        if not os.path.isdir(curr):
            continue
            
        subdirs = [d for d in os.listdir(curr) if os.path.isdir(os.path.join(curr, d))]
        subdir_names_lower = {d.lower() for d in subdirs}
        
        # If we see 3 or more class matches, this is our root
        if len(subdir_names_lower.intersection(target_classes)) >= 3:
            return curr
            
        for d in subdirs:
            queue.append(os.path.join(curr, d))
            
    return base_path

def train():
    # Setup directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # Search for dataset root
    local_path = 'data/Tobacco3482'
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Dataset directory not found at '{local_path}'. Please copy it there.")
        
    root_dir = find_dataset_root(local_path)
    print(f"Detected dataset root directory: {root_dir}")
    
    # Class names are subdirectories
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"Detected classes: {class_names}")
    
    # Save class mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
        
    # Gather image samples
    samples = []
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                fpath = os.path.normpath(os.path.join(class_path, fname))
                samples.append((fpath, label_idx))
                
    print(f"Total images gathered: {len(samples)}")
    if len(samples) == 0:
        raise ValueError(f"No valid images found in root '{root_dir}'. Please verify the folder contents.")
        
    # Stratified Split: 80% train, 10% val, 10% test
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    
    # Train+Val (90%) and Test (10%)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Train (80%) and Val (10%) -> Val is 11.11% of Train+Val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.1111, random_state=42, stratify=train_val_labels
    )
    
    # Save split configuration
    splits = {
        'train': list(zip(train_paths, train_labels)),
        'val': list(zip(val_paths, val_labels)),
        'test': list(zip(test_paths, test_labels))
    }
    with open('data/splits.json', 'w') as f:
        json.dump(splits, f, indent=4)
    print("Saved dataset split configuration to data/splits.json")
    
    # Create PyTorch datasets
    train_transform, val_transform = get_transforms()
    train_dataset = TobaccoDataset(splits['train'], transform=train_transform)
    val_dataset = TobaccoDataset(splits['val'], transform=val_transform)
    
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(splits['test'])}")
    
    # Loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize MobileNetV2 with pre-trained weights
    model = get_model(num_classes=len(class_names), pretrained=True)
    
    # Freeze the first 14 layers in the feature extractor
    print("Freezing early layers (0 to 13) of MobileNetV2 feature extractor...")
    for i in range(14):
        for param in model.features[i].parameters():
            param.requires_grad = False
            
    model = model.to(device)
    
    # Calculate class weights for unbalanced classes
    from collections import Counter
    train_labels = [s[1] for s in splits['train']]
    counter = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_names)
    class_weights = [total_samples / (num_classes * counter[i]) for i in range(num_classes)]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Calculated class weights: {class_weights}")
    
    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # We only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training Loop
    epochs = 12
    best_val_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}")
        
        # Save best model weights
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'models/document_classifier.pth')
            print(f"--> Saved best model weights with Val Acc: {best_val_acc:.4f}")
            
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    print("Training completed! Best Validation Accuracy: {:.4f}".format(best_val_acc))

if __name__ == '__main__':
    train()
