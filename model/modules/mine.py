import torch.nn as nn
from model.modules.fullyconnected_layer import FullyConnectedLayer

# 1.2 定义神经网络模型
class mine(nn.Module):
    def __init__(self,autoIvDinConfig):
        super(mine, self).__init__()
        embedding_size = autoIvDinConfig.embedding_size
        # self.fc1 = FullyConnectedLayer(input_size=embedding_size,hidden_unit=[200, 10],bias=[True, True],batch_norm=False,sigmoid=True,activation='prelu',dice_dim=2)
        self.fc1 = FullyConnectedLayer(input_size=embedding_size, hidden_unit=[32,16], bias=[True,True],
                                       batch_norm=False, sigmoid=True, activation='prelu', dice_dim=2)

        self.fc2 = self.fc1
        self.fc3 = FullyConnectedLayer(input_size=16,
                                       hidden_unit=[1],
                                       bias=[False],
                                       batch_norm=False,
                                       sigmoid=True,
                                       activation='prelu',
                                       dice_dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=autoIvDinConfig.mine_dropout)

    def forward(self, x, y):
        x_1 = self.fc1(x)
        y_1 = self.fc2(y)

        x_out = self.dropout(x_1)
        y_out = self.dropout(y_1)
        out = x_out + y_out

        h1 = self.relu(out)
        h2 = self.fc3(h1)
        # output
        h2_out = self.dropout(h2)
        return h2_out
