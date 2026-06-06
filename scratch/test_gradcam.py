import torch
import torch.nn as nn
from torchvision import models
import os
import sys

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import get_model, get_transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # Register hooks
        self.handlers.append(target_layer.register_forward_hook(self.save_activation))
        self.handlers.append(target_layer.register_full_backward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = output.argmax(dim=1).item()
            
        loss = output[0, target_category]
        loss.backward()
        
        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients
        
        if activations is None or gradients is None:
            raise RuntimeError("Activations or gradients are None. Hooks might not be registered correctly.")
            
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))[0]
        
        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]
            
        # Apply ReLU (only positive contributions)
        heatmap = torch.clamp(heatmap, min=0)
        
        # Normalize heatmap to [0, 1]
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap.cpu().numpy(), target_category
        
    def remove_hooks(self):
        for h in self.handlers:
            h.remove()

def main():
    print("Testing Grad-CAM integration...")
    # Load class mapping to see count
    num_classes = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Instantiate model
    model = get_model(num_classes=num_classes, pretrained=False)
    
    # Try loading weights if exists
    weights_path = 'models/document_classifier.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print("Warning: Model weights not found. Using random weights.")
        
    model = model.to(device)
    model.eval()
    
    # Get last convolutional layer of MobileNetV2
    # MobileNetV2 has model.features which is sequential, ending with model.features[-1] (Conv2dNormActivation)
    # Let's inspect model.features[-1]
    print(f"Last block in features: {model.features[-1]}")
    target_layer = model.features[-1]
    
    # Initialize Grad-CAM
    cam = GradCAM(model, target_layer)
    
    # Generate a random input tensor
    dummy_input = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
    
    try:
        heatmap, cat = cam(dummy_input, target_category=2)
        print(f"Grad-CAM run successful!")
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Heatmap min/max: {heatmap.min():.4f} / {heatmap.max():.4f}")
        print("Success!")
    except Exception as e:
        print(f"Error during Grad-CAM: {e}")
    finally:
        cam.remove_hooks()

if __name__ == '__main__':
    main()
