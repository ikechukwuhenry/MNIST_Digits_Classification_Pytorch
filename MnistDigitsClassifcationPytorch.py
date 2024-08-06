# !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# !tar -zxvf MNIST.tar.gz

# import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
training_accuracies = []
training_losses = []
num_epochs = 5


# Loading Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="data", download=False, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the image classifier model
class ImageClassifier(nn.Module):

    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
        

# Create an instance of the image classifier model
mps_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classifier = ImageClassifier().to(mps_device)

# Define the optimizer and loss function
optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()



# Train the model
def train():

    for epoch in range(num_epochs): # Train for 10 epochs
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(mps_device), labels.to(mps_device)   # Move the data to the device(CPU or GPU)
            optimizer.zero_grad()           # Reset gradients
            ouputs = classifier(images)     # Forward pass
            _, predicted = torch.max(ouputs, 1) # To use and compute accuracy
            loss = loss_fn(ouputs, labels)  # Compute loss
            loss.backward()                 # Backward pass
            optimizer.step()                # Update weights

            # Update the running total of correct predictions and samples
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
        # Calculate Loss and Accuracy for this epoch   
        print(f"Epoch:{epoch} loss is {loss.item()}")
    
        accuracy = 100 * total_correct / total_samples
        print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')

        training_losses.append(loss.item())
        training_accuracies.append(accuracy)


    # Save the trained model
    torch.save(classifier.state_dict(), 'model_state.pt')

# Load the saved model 
# with open('model_state.pt', 'rb') as f:
#     classifier.load_state_dict(load(f))



    # Plot the accuracy over time(epoch) and also loss function over time(epoch)
    import matplotlib.pyplot as plt

    # Plot accuracy
    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), training_accuracies, label='Accuracy')
    # ax.plot(range(num_epochs), training_losses, label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy and Loss')
    ax.set_title('Accuracy per Epoch')
    # plt.legend() # to annotate multiple lines in the same plot
    plt.show()

    # # plot Losses
    fig2, ax2 = plt.subplots()
    ax2.plot(range(num_epochs), training_losses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss per Epoch')
    plt.show()


if __name__ == "__main__":
    train()