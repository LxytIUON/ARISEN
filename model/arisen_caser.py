import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Arisen_CaserConfig
from dataset.loader import load_pretrained_embedding,load_corresponding_iv
from model.modules.mlp import MLP
from model.modules.mine import mine
from model.modules.fullyconnected_layer import FullyConnectedLayer

embedding_size = Arisen_CaserConfig.embedding_size

class Arisen_Caser(nn.Module):
    def __init__(self):
        super(IV_Caser,self).__init__()
        self.name = Arisen_CaserConfig.description
        self.device = Arisen_CaserConfig.device
        self.regularization_weight = []

        # init args
        self.max_seq_length = 8
        self.embedding_size = Arisen_CaserConfig.embedding_size
        self.n_h = 16
        self.n_v = 4
        self.drop_ratio = 0.5
        self.ac_conv = F.relu
        self.ac_fc = F.relu

        # user and item embeddings
        item_embedding_matrix, post_embedding_matrix = load_pretrained_embedding()
        self.item_embedding_layer = nn.Embedding.from_pretrained(item_embedding_matrix, freeze=True)
        self.post_embedding_layer = nn.Embedding.from_pretrained(post_embedding_matrix, freeze=True)
        _,_,query_embedding_matrix, cor_query_matrix = load_corresponding_iv()
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_matrix, freeze=True)
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

        # MLP1 and MLP2
        self.alpha = MLP(Arisen_CaserConfig.alpha)
        self.add_regularization_weight(self.alpha.parameters(), l2=Arisen_CaserConfig.l2_lambda)
        self.beta = MLP(Arisen_CaserConfig.beta)
        self.add_regularization_weight(self.beta.parameters(), l2=Arisen_CaserConfig.l2_lambda)

        if Arisen_CaserConfig.connection_type == "z":
            self.bridge1 = torch.nn.Linear(embedding_size, embedding_size)
            torch.nn.init.orthogonal_(self.bridge1.weight)
            self.bridge2 = torch.nn.Linear(embedding_size, embedding_size)
            torch.nn.init.orthogonal_(self.bridge2.weight)

        self.mine = mine(Arisen_CaserConfig)

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

    def iv(self, item, query, query_pinv, y):
        '''
        input:
            item:<batch,history_size/1,embedding_size>
            query:<batch,history_size/1,embedding_size,1>
            query_pinv:<batch,history_size/1,1,embedding_size>
        output：
            reconstruct_embedding:<batch,history_size,embedding_size>
        '''
        x = item

        query_origin = query.flatten(start_dim=2)  # <batch,history_size/1,embedding_size>

        alpha = self.prob_sigmoid(self.alpha(torch.cat([item, query_origin], dim=-1)))  # <batch,history_size/1,1>
        beta = self.prob_sigmoid(self.beta(torch.cat([item, query_origin], dim=-1)))  # <batch,history_size/1,1>

        item = item.unsqueeze(dim=-1)  # <batch,history_size/1,embedding_size,1>
        t_1 = torch.matmul(query, torch.matmul(query_pinv, item))  # <batch,history_size/1,embedding_size,1>

        z = torch.matmul(t_1, alpha.unsqueeze(dim=-1))  # <batch,history_size/1,embedding_size,1>
        c = torch.matmul(t_1, beta.unsqueeze(dim=-1))# <batch,history_size/1,embedding_size,1>

        return z,c,x,y

    def calculate_mi(self,e_z, e_c, e_x, e_y, shape):
        # 计算互信息
        item_e_z = torch.mean(e_z, dim=1).reshape(shape, embedding_size)
        item_e_c = torch.mean(e_c, dim=1).reshape(shape, embedding_size)
        item_e_x = torch.mean(e_x, dim=1).reshape(shape,embedding_size)  # <batch,history_size,embedding_size>  ->  <batch,1,embedding_size>
        item_e_y = e_y.reshape(shape, embedding_size)

        loss_zx = torch.mean(self.mine(item_e_z, item_e_x)) - torch.log(torch.mean(torch.exp(self.mine(item_e_z, item_e_x[torch.randperm(shape)]))))
        loss_zy = torch.mean(self.mine(item_e_z, item_e_y)) - torch.log(torch.mean(torch.exp(self.mine(item_e_z, item_e_y[torch.randperm(shape)]))))
        loss_cx = torch.mean(self.mine(item_e_c, item_e_x)) - torch.log(torch.mean(torch.exp(self.mine(item_e_c, item_e_x[torch.randperm(shape)]))))
        loss_cy = torch.mean(self.mine(item_e_c, item_e_y)) - torch.log(torch.mean(torch.exp(self.mine(item_e_c, item_e_y[torch.randperm(shape)]))))
        loss_zc = torch.mean(self.mine(item_e_z, item_e_c)) - torch.log(torch.mean(torch.exp(self.mine(item_e_z, item_e_c[torch.randperm(shape)]))))

        mi_loss = - loss_zx + loss_zy - 1.0 * (loss_cx + loss_cy) + 0.1 * loss_zc

        return mi_loss
    
    def forward(self, x):
        x = [i.to(self.device) for i in x]
        _,click_item, target_item, click_post, query = x

        # Embedding
        click_embedding = self.embedder(self.item_embedding_layer(click_item)) 
        item_embedding = self.embedder(self.item_embedding_layer(target_item))
        post_embedding = self.embedder(self.post_embedding_layer(click_post))
        query_embedding = self.embedder(self.query_embedding_layer(query)).reshape(click_post.shape[0],click_post.shape[1], embedding_size,1)
        cor_query_embedding = self.embedder(self.cor_query_pinv(query)).reshape(click_post.shape[0],click_post.shape[1], 1, embedding_size)
        e_z, e_c, e_x, e_y = self.iv(post_embedding, query_embedding, cor_query_embedding,y=item_embedding)  # <batch, history_size, embedding_size>
        post_embedding = e_z.squeeze_(dim=-1) + e_c.squeeze_(dim=-1) 

        click_embedding = torch.concat((click_embedding ,post_embedding),1).unsqueeze(1)

        mi_loss = self.calculate_mi(e_z, e_c, e_x, e_y, target_item.shape[0])

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(click_embedding)
            # print(out_v.shape)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect # [2944, 256]
            # print(out_v.shape)

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(click_embedding).squeeze(3))
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

        concat_embedding = torch.concat((z,item_embedding),1)
        # output
        pred = self.prob_sigmoid(self.fc2(concat_embedding))

        return pred.squeeze(1), mi_loss
