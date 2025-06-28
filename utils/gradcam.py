import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        # Ensure input_tensor requires gradients
        input_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model(pixel_values=input_tensor)
        logits = outputs.logits
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        
        # Get the score for the target class
        score = logits[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Get the gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Create CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Normalize CAM
        cam = np.maximum(cam.cpu().numpy(), 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam) + 1e-8
        
        return cam
