import torch
import torch.nn as nn


class EncResUnit(nn.Module):
    def __init__(self, channels, features, stride):
        super().__init__()
        self._c = channels
        self._f = features
        self._s = stride

        self.conv1 = nn.Conv2d(self._c, self._f, self._s+1*2, self._s, 1, padding_mode="replicate")
        self.relu = nn.PReLU(self._f)
        self.conv2 = nn.Conv2d(self._f, self._c, 3, 1, 1, padding_mode="replicate")
        if self._s > 1:
            self.down = nn.AvgPool2d(self._s)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)

        if self._s == 1:
            z = x
        else:
            z = self.down(x)

        return y + z


class RefineNet(nn.Module):
    """
    A convolutional neural network designed for refining features or image data.
    It employs several residual units (EncResUnit) between convolutional layers.
    """
    def __init__(self, in_channels: int, latent_channels: int, out_channels: int):
        """
        Initializes the RefineNet.

        Args:
            in_channels (int): Number of input channels.
            latent_channels (int): Number of channels in the intermediate layers
                                   and residual units.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.out_channels = out_channels

        # Ensure EncResUnit is defined/imported
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, self.latent_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            EncResUnit(self.latent_channels, self.latent_channels, stride=1),
            EncResUnit(self.latent_channels, self.latent_channels, stride=1),
            EncResUnit(self.latent_channels, self.latent_channels, stride=1),
            EncResUnit(self.latent_channels, self.latent_channels, stride=1),
            nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1), # Typically 1x1 conv for final channel adjustment
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the refinement network.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing by the network.
        """
        return self.model(inputs)