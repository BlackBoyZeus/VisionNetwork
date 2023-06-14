import torch
import torch.nn as nn
import torchvision.models as models

class VisionModel(nn.Module):
    """
    A base class for vision models that encapsulates common functionality.
    """

    def __init__(self, num_classes):
        super(VisionModel, self).__init__()
        self.num_classes = num_classes
        self.features = None
        self.classifier = None
        self._initialize_model()

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    def _initialize_model(self):
        raise NotImplementedError("Subclasses must implement _initialize_model method.")

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = True

class ResNetModel(VisionModel):
    """
    A ResNet-based vision model with customizable depth and number of classes.
    """

    def __init__(self, depth, num_classes):
        self.depth = depth
        super(ResNetModel, self).__init__(num_classes)

    def _initialize_model(self):
        if self.depth == 18:
            resnet = models.resnet18(pretrained=True)
        elif self.depth == 34:
            resnet = models.resnet34(pretrained=True)
        elif self.depth == 50:
            resnet = models.resnet50(pretrained=True)
        elif self.depth == 101:
            resnet = models.resnet101(pretrained=True)
        elif self.depth == 152:
            resnet = models.resnet152(pretrained=True)
        else:
            raise ValueError("Invalid ResNet depth specified.")

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(resnet.fc.in_features, self.num_classes)

# Additional vision models can be added as needed

# In this example, the `vision_models.py` file defines a base class `VisionModel` that provides common functionality for vision models.
# It includes methods for forwarding inputs through the model, initializing the model architecture, and freezing/unfreezing the model's features for transfer learning.

# The file also includes an example subclass `ResNetModel`, which is a ResNet-based vision model.
# It allows customization of the ResNet depth (e.g., 18, 34, 50, 101, 152) and the number of output classes.
# The `_initialize_model` method is overridden in the subclass to initialize the ResNet architecture with the specified depth and pretrained weights.

# You can extend this file by adding more vision model subclasses or introducing other modular components that suit your needs.
# Remember to customize the models and their architectures based on your specific requirements and consider using appropriate activation functions, regularization techniques, or additional layers as needed.

# This example aims to provide a starting point for building versatile, transferable, modular, flexible, thoughtful, and original vision models.
# Feel free to further modify and expand upon it to meet your specific use case and incorporate any additional functionality or improvements you desire.
