import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioModel(nn.Module):
    """
    A simple CNN-based audio classification model for voice intent recognition.
    """
    def __init__(self, num_classes=10, input_channels=1, sample_rate=16000):
        """
        Initialize the AudioModel.
        
        Args:
            num_classes (int): Number of output classes for classification
            input_channels (int): Number of input channels (1 for mono, 2 for stereo)
            sample_rate (int): Sample rate of the input audio (used for some layer configurations)
        """
        super(AudioModel, self).__init__()
        self.sample_rate = sample_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=80, stride=4, padding=38)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate the size of the flattened features
        # This will be updated in the first forward pass
        self._flattened_size = None
        
        # Fully connected layers
        # We'll initialize these in the first forward pass
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input shape: (batch_size, channels, time_steps)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten the features
        if self._flattened_size is None:
            self._flattened_size = x.shape[1] * x.shape[2]
            # Initialize fully connected layers now that we know the input size
            self.fc1 = nn.Linear(self._flattened_size, 512).to(x.device)
            self.fc2 = nn.Linear(512, 256).to(x.device)
            self.fc3 = nn.Linear(256, self.num_classes).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def create_model(config):
    """
    Factory function to create an instance of AudioModel with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing model parameters
        
    Returns:
        AudioModel: An instance of the AudioModel
    """
    model_config = config.get('model', {})
    num_classes = model_config.get('num_classes', 10)
    input_channels = model_config.get('input_channels', 1)
    sample_rate = model_config.get('sample_rate', 16000)
    
    return AudioModel(
        num_classes=num_classes,
        input_channels=input_channels,
        sample_rate=sample_rate
    )
