import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)#RGB, change it to Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5) 

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.to_features = nn.Linear(20 * 4 * 4, 128)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )

    def forward(self, x, hidden=None):
        # x shape: [batch, time, channels, height, width]
        batch, time, channels, height, width = x.shape

        # Process each frame with the CNN.
        x = x.reshape(batch * time, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.to_features(x))

        # Restore the time dimension for the GRU.
        x = x.reshape(batch, time, 128)
        sequence, hidden = self.gru(x, hidden)

        return sequence, hidden