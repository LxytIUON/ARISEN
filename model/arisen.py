import torch
import torch.nn as nn
from model.modules.mlp import MLP
from model.modules.fullyconnected_layer import FullyConnectedLayer
from model.modules.attention_pooling_layer import AttentionSequencePoolingLayer
from model.modules.mine import mine
from dataset.loader import load_pretrained_embedding,load_corresponding_iv,load_pretrained_user
from config.config import ArisenConfig

embedding_size = ArisenConfig.embedding_size

class Arisen(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = ArisenConfig.description
        self.device = ArisenConfig.device

        self.regularization_weight = []

        item_embedding_matrix, post_embedding_matrix = load_pretrained_embedding()

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_embedding_matrix, freeze=True)
        self.post_embedding_layer = nn.Embedding.from_pretrained(post_embedding_matrix, freeze=True)

        _,_,query_embedding_matrix, cor_query_matrix = load_corresponding_iv()
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_matrix, freeze=True)


        # MLP1 and MLP2
        self.alpha = MLP(ArisenConfig.alpha)
        self.add_regularization_weight(self.alpha.parameters(), l2=ArisenConfig.l2_lambda)
        self.beta = MLP(ArisenConfig.beta)
        self.add_regularization_weight(self.beta.parameters(), l2=ArisenConfig.l2_lambda)

        # attention层
        self.attn1 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.attn2 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)

        # 全连接层
        self.fc_layer = FullyConnectedLayer(input_size=3 * embedding_size,
                                            hidden_unit=[64, 16, 1],
                                            bias=[True, True, False],
                                            batch_norm=False,
                                            sigmoid=True,
                                            activation='relu',
                                            dice_dim=3)

        if ArisenConfig.connection_type == "z":
            self.bridge1 = torch.nn.Linear(embedding_size, embedding_size)
            torch.nn.init.orthogonal_(self.bridge1.weight)
            self.bridge2 = torch.nn.Linear(embedding_size, embedding_size)
            torch.nn.init.orthogonal_(self.bridge2.weight)

        self.prob_sigmoid = nn.Sigmoid()

        self.mine = mine(ArisenConfig)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
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
            item:<batch,10/1,embedding_size>
            query:<batch,10/1,embedding_size,1>
            query_pinv:<batch,10/1,1,embedding_size>
        output：
            reconstruct_embedding:<batch,40,embedding_size>
        '''
        x = item
        query_origin = query.flatten(start_dim=2)  # <batch,10/1,embedding_size>

        alpha = self.prob_sigmoid(self.alpha(torch.cat([item, query_origin], dim=-1)))  # <batch,40/1,1>
        beta = self.prob_sigmoid(self.beta(torch.cat([item, query_origin], dim=-1)))  # <batch,40/1,1>

        item = item.unsqueeze(dim=-1)  # <batch,40/1,embedding_size,1>
        t_1 = torch.matmul(query, torch.matmul(query_pinv, item))  # <batch,40/1,embedding_size,1>

        z = torch.matmul(t_1, alpha.unsqueeze(dim=-1))  # <batch,40/1,embedding_size,1>
        c = torch.matmul(t_1, beta.unsqueeze(dim=-1))  # <batch,40/1,embedding_size,1>

        return z, c, x, y

    def calculate_mi(self,e_z, e_c, e_x, e_y, shape):
        # 计算互信息
        item_e_z = torch.mean(e_z, dim=1).reshape(shape, embedding_size)
        item_e_c = torch.mean(e_c, dim=1).reshape(shape, embedding_size)
        item_e_x = torch.mean(e_x, dim=1).reshape(shape,embedding_size)  # <batch,40,embedding_size>  ->  <batch,1,embedding_size>
        item_e_y = e_y.reshape(shape, embedding_size)

        loss_zx = torch.mean(self.mine(item_e_z, item_e_x)) - torch.log(torch.mean(torch.exp(self.mine(item_e_z, item_e_x[torch.randperm(shape)]))))
        loss_zy = torch.mean(self.mine(item_e_z, item_e_y)) - torch.log(torch.mean(torch.exp(self.mine(item_e_z, item_e_y[torch.randperm(shape)]))))
        loss_cx = torch.mean(self.mine(item_e_c, item_e_x)) - torch.log(torch.mean(torch.exp(self.mine(item_e_c, item_e_x[torch.randperm(shape)]))))
        loss_cy = torch.mean(self.mine(item_e_c, item_e_y)) - torch.log(torch.mean(torch.exp(self.mine(item_e_c, item_e_y[torch.randperm(shape)]))))
        loss_zc = torch.mean(self.mine(item_e_z, item_e_c)) - torch.log(torch.mean(torch.exp(self.mine(item_e_z, item_e_c[torch.randperm(shape)]))))

        mi_loss = - loss_zx + loss_zy - 1.0 * (loss_cx + loss_cy) + 0.1 * loss_zc

        return mi_loss

    def forward(self, x):

        '''
        input:
            search_log : <batch, query_history>
            behavior_history : <batch, hist_number>
            thread_click_item : <batch, query_history>
            item : <batch>
        '''
        x = [i.to(self.device) for i in x]
        user,click_item, item, click_post, query = x
        
        item_embedding = self.embedder(self.item_embedding_layer(item))  # <batch, embedding_size>
        post_embedding = self.embedder(self.post_embedding_layer(click_post))
        query_embedding = self.embedder(self.query_embedding_layer(query)).reshape(click_post.shape[0], click_post.shape[1], embedding_size, 1)
        cor_query_embedding = self.embedder(self.cor_query_pinv(query)).reshape(click_post.shape[0], click_post.shape[1], 1, embedding_size)
        click_embedding = self.embedder(self.item_embedding_layer(click_item))

        # 解耦社区序列
        e_z, e_c, e_x, e_y = self.iv(post_embedding, query_embedding, cor_query_embedding,y=item_embedding)  # <batch, 40, embedding_size>
        post_embedding = e_z.squeeze_(dim=-1) + e_c.squeeze_(dim=-1)# 解耦后的社区序列部分
        
        post_mask = torch.where(click_post == 0, 1, 0).bool()  # <batch,40>
        click_mask = torch.where(click_item == 0, 1, 0).bool()

        # 计算互信息
        mi_loss = self.calculate_mi(e_z, e_c, e_x, e_y, item.shape[0])
        
        # 正交矩阵
        if ArisenConfig.connection_type == "z":
            click_embedding = self.bridge1(click_embedding)  # <batch, 50, embedding_size>
            post_embedding = self.bridge2(post_embedding)  # <batch, 10, embedding_size>
            
        # concat
        if ArisenConfig.connection_type == "concat":
            pass

        # underlying model：attention-concat-fc
        # attention
        item_embedding = item_embedding.reshape(item.shape[0], 1, embedding_size)
        click_atten = self.attn1(item_embedding, click_embedding, click_mask)  # <batch,1,embedding_size>
        post_atten = self.attn2(item_embedding, post_embedding, post_mask)  # <batch,1,embedding_size>
        
        # concat
        concat_feature = torch.cat([item_embedding, click_atten, post_atten],dim=-1)  # <batch,1,embedding_size*3>
        
        # fc# output
        output = self.fc_layer(concat_feature)  # <batch,1,1>

        return output.squeeze(1).squeeze(1), mi_loss
