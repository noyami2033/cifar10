import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        def conv(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.feature_extractor = nn.Sequential(
            conv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            conv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            conv(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
            )

        self.num_classes = 10
        self.flatten_size = 64*32*32
        self.fc = nn.Linear(self.flatten_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# #test
# model = Network()
# print(model)
# dummy_input = torch.ones(16, 3, 32, 32) #(batch_size, channel, height, width)
# dummy_output = model(dummy_input)
# print(dummy_output.shape)