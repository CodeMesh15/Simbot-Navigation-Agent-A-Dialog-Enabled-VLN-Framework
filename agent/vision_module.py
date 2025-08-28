
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

class VisionModule:
    """
    Processes visual input from the environment using a pre-trained CNN.
    """
    def __init__(self, device):
        self.device = device
        
        # Load a pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the model to be a feature extractor by removing the final classification layer
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Set the model to evaluation mode and move it to the specified device
        self.model.eval()
        self.model.to(self.device)
        
        # Define the image preprocessing steps
        self.preprocess = T.Compose([
            T.ToPILImage(), # Convert numpy array to PIL Image
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad() # Disable gradient calculations for inference
    def extract_features(self, image_np):
        """
        Takes a raw image (numpy array) and returns a feature vector.
        
        Args:
            image_np (np.ndarray): The input image from the simulator (H, W, C).
            
        Returns:
            torch.Tensor: A feature vector of shape (1, 2048).
        """
        # Preprocess the image and add a batch dimension
        input_tensor = self.preprocess(image_np).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Pass the image through the model
        features = self.model(input_tensor)
        
        # Flatten the output to a 1D vector
        return features.squeeze()

if __name__ == '__main__':
    # --- Example Usage ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_processor = VisionModule(device)
    
    # Create a dummy image (e.g., from the AI2-THOR simulator)
    import numpy as np
    dummy_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    
    print("Extracting features from a dummy image...")
    features = vision_processor.extract_features(dummy_image)
    
    print(f"Vision Module loaded successfully on '{device}'.")
    print(f"Output feature vector shape: {features.shape}")
