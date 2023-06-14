import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10

class VisionTrainer:
    """
    A trainer class for training a vision network.
    """

    def __init__(self, model, train_dataset, test_dataset, batch_size, learning_rate):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs):
        """
        Train the vision network for the specified number of epochs.

        Args:
            num_epochs (int): The number of training epochs.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        self.save_model()

    def test(self):
        """
        Evaluate the trained model on the test dataset and compute accuracy.
        """
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

    def save_model(self):
        """
        Save the trained model.
        """
        torch.save(self.model.state_dict(), "trained_model.pt")

# Example usage
train_dataset = CIFAR10(root="./data", train=True, transform=ToTensor(), download=True)
test_dataset = CIFAR10(root="./data", train=False, transform=ToTensor(), download=True)

model = YourVisionModel()  # Replace with your own vision model
trainer = VisionTrainer(model, train_dataset, test_dataset, batch_size=64, learning_rate=0.001)
trainer.train(num_epochs=10)
trainer.test()

# In this example, the `VisionTrainer` class handles the training and evaluation of the vision network.
# It takes in the model, training and test datasets, batch size, and learning rate as inputs.
# The `train` method performs the training loop, while the `test` method evaluates the trained model on the test dataset.
# The `save_model` method saves the trained model to a file.

# You can customize this file to fit your specific vision network architecture and requirements.
# For example, you can modify the data loading process, incorporate additional evaluation metrics,
# or implement advanced training techniques like learning rate scheduling or early stopping.
