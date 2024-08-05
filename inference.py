# import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from MnistDigitsClassifcationPytorch import ImageClassifier

# Create an instance of the image classifier model
mps_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classifier = ImageClassifier().to(mps_device)

# Load the saved model 
with open('model_state.pt', 'rb') as f:
    classifier.load_state_dict(load(f))

classifier.eval()

# Perform inference on an image
img = Image.open('image.jpg')
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img).unsqueeze(0).to(mps_device)
output = classifier(img_tensor)
predicted_label = torch.argmax(output)
print(f"Predicted label: {predicted_label}")