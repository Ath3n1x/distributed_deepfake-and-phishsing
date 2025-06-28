import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

def create_model(device='cpu', pretrained=True, quantize=False):
    # Load the pre-trained model and processor
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    
    # Only move to device if CUDA is available and requested
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
    
    if quantize:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    return model, processor
