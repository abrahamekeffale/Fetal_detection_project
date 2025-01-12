import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        
        super(SegNet, self).__init__()

        # Encoder layers
        self.encoder_conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder layers
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder_conv1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x, indices1 = self.pool1(x)

        x = F.relu(self.encoder_conv3(x))
        x = F.relu(self.encoder_conv4(x))
        x, indices2 = self.pool2(x)

        x = F.relu(self.encoder_conv5(x))
        x = F.relu(self.encoder_conv6(x))
        x, indices3 = self.pool3(x)

        # Decoder
        x = self.unpool3(x, indices3)
        x = F.relu(self.decoder_conv6(x))
        x = F.relu(self.decoder_conv5(x))

        x = self.unpool2(x, indices2)
        x = F.relu(self.decoder_conv4(x))
        x = F.relu(self.decoder_conv3(x))

        x = self.unpool1(x, indices1)
        x = F.relu(self.decoder_conv2(x))
        x = self.decoder_conv1(x)

        return torch.sigmoid(x)  # Use sigmoid for binary segmentation

# Initialize the SegNet model for grayscale images
input_channels = 1  # Grayscale images
num_classes = 1     # Binary segmentation
model = SegNet(input_channels=input_channels, num_classes=num_classes)

