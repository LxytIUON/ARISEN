import torch.nn as nn
import torch

class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input.
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        # output
        return out


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, bias, batch_norm=False, activation='relu', sigmoid=False, dice_dim=2):
        super(FullyConnectedLayer, self).__init__()

        assert len(hidden_unit) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_unit)

        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0], bias=bias[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_unit[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i + 1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)

        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)