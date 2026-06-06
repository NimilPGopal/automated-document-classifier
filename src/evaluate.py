import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from utils import TobaccoDataset, get_model, get_transforms

def evaluate():
    os.makedirs('reports', exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")
    
    # Load class mapping
    if not os.path.exists('models/class_mapping.json'):
        raise FileNotFoundError("Class mapping file not found. Please run train.py first.")
        
    with open('models/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    # class_mapping keys are strings due to JSON format, map to integers
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    
    # Load test split
    if not os.path.exists('data/splits.json'):
        raise FileNotFoundError("Split configuration file not found in data/splits.json. Run train.py first.")
        
    with open('data/splits.json', 'r') as f:
        splits = json.load(f)
    test_split = splits['test']
    print(f"Loaded test split with {len(test_split)} items.")
    
    # Wrap test dataset
    _, val_transform = get_transforms()
    test_dataset = TobaccoDataset(test_split, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load model
    model = get_model(num_classes=len(class_names), pretrained=False)
    if not os.path.exists('models/document_classifier.pth'):
        raise FileNotFoundError("Model file not found. Please run train.py first.")
        
    model.load_state_dict(torch.load('models/document_classifier.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Classification report
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n", report_text)
    
    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix - Tobacco3482 Classifier')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved confusion matrix heatmap to reports/confusion_matrix.png")
    
    # 3. Precision-Recall Curve (for each class)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (all_labels == i).astype(int)
        y_scores = all_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        ap = average_precision_score(y_true_binary, y_scores)
        
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.4f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Tobacco3482 Classifier')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/precision_recall.png', dpi=300)
    plt.close()
    print("Saved Precision-Recall curves to reports/precision_recall.png")
    
    # 4. Generate evaluation_report.md
    history_md = ""
    if os.path.exists('models/training_history.json'):
        with open('models/training_history.json', 'r') as f:
            history = json.load(f)
        
        # Plot training curves and save
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('reports/training_curves.png', dpi=300)
        plt.close()
        print("Saved training curves plot to reports/training_curves.png")
        
        # Format training history in markdown
        history_md = f"""
## 📈 Training History

Here is a summary of the model training over epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| :---: | :---: | :---: | :---: | :---: |
"""
        for epoch in range(len(history['train_loss'])):
            history_md += f"| {epoch+1} | {history['train_loss'][epoch]:.4f} | {history['train_acc'][epoch]:.4f} | {history['val_loss'][epoch]:.4f} | {history['val_acc'][epoch]:.4f} |\n"
            
        history_md += "\n![Training Curves](training_curves.png)\n"

    report_md = f"""# 📊 Tobacco3482 Document Classifier Evaluation Report

This report presents the metrics and charts evaluating the trained MobileNetV2 document classifier.

## 🏆 Summary Metrics

- **Test Accuracy**: {report_dict['accuracy']:.4f}
- **Macro Average F1-score**: {report_dict['macro avg']['f1-score']:.4f}
- **Weighted Average F1-score**: {report_dict['weighted avg']['f1-score']:.4f}

---

{history_md}

---

## 📋 Detailed Classification Report

Below is the classification report showing Precision, Recall, and F1-score for each of the 10 classes.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for class_name in class_names:
        c_metrics = report_dict[class_name]
        report_md += f"| **{class_name}** | {c_metrics['precision']:.4f} | {c_metrics['recall']:.4f} | {c_metrics['f1-score']:.4f} | {int(c_metrics['support'])} |\n"
        
    report_md += f"| | | | | |\n"
    report_md += f"| **Accuracy** | | | {report_dict['accuracy']:.4f} | {int(report_dict['macro avg']['support'])} |\n"
    report_md += f"| **Macro Avg** | {report_dict['macro avg']['precision']:.4f} | {report_dict['macro avg']['recall']:.4f} | {report_dict['macro avg']['f1-score']:.4f} | {int(report_dict['macro avg']['support'])} |\n"
    report_md += f"| **Weighted Avg** | {report_dict['weighted avg']['precision']:.4f} | {report_dict['weighted avg']['recall']:.4f} | {report_dict['weighted avg']['f1-score']:.4f} | {int(report_dict['weighted avg']['support'])} |\n"

    report_md += """
---

## 🗺️ Confusion Matrix

The confusion matrix shows the true classes versus predicted classes. This helps analyze where the model confuses documents (for example, Letters vs. Memos).

![Confusion Matrix](confusion_matrix.png)

---

## 📈 Precision-Recall Curves

Precision-Recall curves measure the tradeoff between precision and recall for different thresholds. The higher the Area Under Precision-Recall curve (Average Precision, AP), the better the classifier performs on that category.

![Precision-Recall Curves](precision_recall.png)
"""

    with open('reports/evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print("Saved evaluation report markdown to reports/evaluation_report.md")

if __name__ == '__main__':
    evaluate()
