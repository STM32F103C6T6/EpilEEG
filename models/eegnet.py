# models/eegnet.py
# Example implementation (refer to original EEGNet paper/code for details)
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_classes, chans, samples, dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, **kwargs):  # Accept extra args
        super(EEGNet, self).__init__()
        # Implementation based on https://github.com/vlawhern/arl-eegmodels
        # Input shape: (batch, 1, chans, samples) -> Add the '1' dimension in predictor/dataloader

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise Conv
            nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.block2 = nn.Sequential(
            # Separable Conv
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 16 // 2), groups=F1 * D, bias=False),  # Depthwise
            nn.Conv2d(F2, F2, (1, 1), bias=False),  # Pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        # Calculate flattened features size dynamically (crucial!)
        # Need to pass dummy data through block1 and block2 to find the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, chans, samples)
            out_block1 = self.block1(dummy_input)
            out_block2 = self.block2(out_block1)
            self.flattened_size = out_block2.cpu().data.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flattened_size, n_classes)

    def forward(self, x):
        # Ensure input has shape (batch, 1, chans, samples)
        if x.dim() == 3:  # If input is (batch, chans, samples)
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
