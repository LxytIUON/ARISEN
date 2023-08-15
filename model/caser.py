import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import CaserConfig
from dataset.loader import load_pretrained_embedding
from model.modules.fullyconnected_layer import FullyConnectedLayer

# CASER
class Caser(nn.Module):
    def __init__(self):
        super(Caser, self).__init__()
        self.name = 'Caser'
        self.device = CaserConfig.device
        self.regularization_weight = []

        # init args
        self.max_seq_length = 4
        self.embedding_size = CaserConfig.embedding_size
        self.n_h = 16
        self.n_v = 4
        self.drop_ratio = 0.5
        self.ac_conv = F.relu
        self.ac_fc = F.relu

        # user and item embeddings
        post_embedding_matrix, item_embedding_matrix = load_pretrained_embedding()
        self.item_embedding_layer = nn.Embedding.from_pretrained(item_embedding_matrix, freeze=True)
        self.embedder = FullyConnectedLayer(input_size=768, hidden_unit=[512,256,128,64], bias=[False,False,False,False],batch_norm=False, sigmoid=False, activation='relu', dice_dim=4)


        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, self.embedding_size)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size #256
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        self.cache_x = None

        self.prob_sigmoid = nn.Sigmoid()

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # 对于参数，将其放在一个列表中以保持与get_regularization_loss()兼容
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        '''
            计算正则化损失(l1 or l2)
        '''
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def forward(self, x):
        x = [i.to(self.device) for i in x]
        _,seq_var, target_item = x # [batch, len]   [batch]
        # Embedding Look-up
        item_embs = self.embedder(self.item_embedding_layer(seq_var)).unsqueeze(1)  # use unsqueeze() to get 4-D # [batch, 1, len, emb]

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            # print(out_v.shape)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect # [2944, 256]
            # print(out_v.shape)

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                # print(conv_out.shape)
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                # print(pool_out.shape)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
        # print(out_h.shape)

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))

        target_item_embedding = self.embedder(self.item_embedding_layer(target_item))

        concat_embedding = torch.concat((z,target_item_embedding),1)

        pred = self.prob_sigmoid(self.fc2(concat_embedding))

        return pred.squeeze(1)
