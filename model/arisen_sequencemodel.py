import torch
import torch.nn as nn
from model.modules.mlp import MLP
from model.modules.fullyconnected_layer import FullyConnectedLayer
from model.modules.attention_pooling_layer import AttentionSequencePoolingLayer
from config.config import Arisen_SequenceModelConfig
from model.modules.mine import mine
from dataset.loader import load_pretrained_embedding,load_corresponding_iv

embedding_size = Arisen_SequenceModelConfig.embedding_size

class Arisen_SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'IV_SequenceModel'
        self.device = Arisen_SequenceModelConfig.device

        self.type = Arisen_SequenceModelConfig.seq_type

        self.regularization_weight = []
        item_embedding_matrix, post_embedding_matrix = load_pretrained_embedding()

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_embedding_matrix, freeze=True)
        self.post_embedding_layer = nn.Embedding.from_pretrained(post_embedding_matrix, freeze=True)
        _,_,query_embedding_matrix, cor_query_matrix = load_corresponding_iv()
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_matrix, freeze=True)

        self.embedder = FullyConnectedLayer(input_size=768, hidden_unit=[512, 256, 128, embedding_size],
                                            bias=[False, False, False, False], batch_norm=False, sigmoid=False,
                                            activation='relu', dice_dim=4)

        # MLP0
        self.mlp = MLP(Arisen_SequenceModelConfig.mlp)
        self.add_regularization_weight(self.mlp.parameters(), l2=Arisen_SequenceModelConfig.l2_lambda)

        # MLP1 and MLP2
        self.alpha = MLP(Arisen_SequenceModelConfig.alpha)
        self.add_regularization_weight(self.alpha.parameters(), l2=Arisen_SequenceModelConfig.l2_lambda)
        self.beta = MLP(Arisen_SequenceModelConfig.beta)
        self.add_regularization_weight(self.beta.parameters(), l2=Arisen_SequenceModelConfig.l2_lambda)

        # attention层
        self.attn1 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.attn2 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)

        if self.type == "GRU":
            self.bridge_layer1 = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, num_layers=2,batch_first=True)
            self.bridge_layer2 = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, num_layers=2,batch_first=True)
        elif self.type == "Transformer":
            self.bridge_layer1 = nn.Transformer(d_model=embedding_size, num_encoder_layers=3,num_decoder_layers=3,dim_feedforward=embedding_size*2,batch_first=True)
            self.bridge_layer2 = nn.Transformer(d_model=embedding_size, num_encoder_layers=3,num_decoder_layers=3,dim_feedforward=embedding_size*2,batch_first=True)

        self.fc_layer = FullyConnectedLayer(input_size=3 * embedding_size,hidden_unit=[200, 80, 1],bias=[True, True, False],batch_norm=False,sigmoid=True,activation='prelu',dice_dim=3)

        self.mine = mine(Arisen_SequenceModelConfig)
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

    def get_src(self,behavior, embedding):
        src = torch.zeros(embedding.shape[0], embedding.shape[1] - 1, embedding.shape[-1])
        for batch in range(embedding.shape[0]):
            for i in range(embedding.shape[1]):
                if behavior[batch][i] == 0:
                    src[batch] = torch.cat((embedding[batch][:i - 1], embedding[batch][i:]), 0)
        return src.to(self.device)

    def iv(self, item, query, query_pinv, y):
        '''
        input:
            item:<batch,40/1,embedding_size>
            query:<batch,40/1,embedding_size,1>
            query_pinv:<batch,40/1,1,embedding_size>
        output：
            reconstruct_embedding:<batch,40,embedding_size>
        '''
        x = item

        query_origin = query.flatten(start_dim=2)  # <batch,40/1,embedding_size>

        alpha = self.prob_sigmoid(self.alpha(torch.cat([item, query_origin], dim=-1)))  # <batch,40/1,1>
        beta = self.prob_sigmoid(self.beta(torch.cat([item, query_origin], dim=-1)))  # <batch,40/1,1>

        item = item.unsqueeze(dim=-1)  # <batch,40/1,embedding_size,1>
        t_1 = torch.matmul(query, torch.matmul(query_pinv, item))  # <batch,40/1,embedding_size,1>

        z = torch.matmul(t_1, alpha.unsqueeze(dim=-1))  # <batch,40/1,embedding_size,1>
        c = torch.matmul(t_1, beta.unsqueeze(dim=-1))# <batch,40/1,embedding_size,1>

        return z,c,x,y

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
        x = [i.to(self.device) for i in x]
        user, user_behavior, item, click_post, query = x

        user_behavior_embedding = self.embedder(self.item_embedding_layer(user_behavior))
        browse_mask = torch.where(user_behavior == 0, 1, 0).bool()

        item_embedding = self.embedder(self.item_embedding_layer(item))
        item_embedding = item_embedding.unsqueeze(1)

        post_embedding = self.embedder(self.post_embedding_layer(click_post))
        orignal_emb = self.post_embedding_layer(click_post)
        post_mask = torch.where(click_post == 0, 1, 0).bool()
        query_embedding = self.embedder(self.query_embedding_layer(query)).reshape(click_post.shape[0],click_post.shape[1], embedding_size,1)
        cor_query_embedding = self.embedder(self.cor_query_pinv(query)).reshape(click_post.shape[0], click_post.shape[1], 1, embedding_size)
        e_z, e_c, e_x, e_y = self.iv(post_embedding, query_embedding, cor_query_embedding,y=item_embedding)  # <batch, 40, embedding_size>
        post_embedding = e_z.squeeze_(dim=-1) + e_c.squeeze_(dim=-1)  # 解耦后的社区序列部分
        mi_loss = self.calculate_mi(e_z, e_c, e_x, e_y, item.shape[0])

        item_embedding = item_embedding.reshape(item.shape[0], 1, embedding_size)
        if self.type == "Transformer":
            user_behavior_bridge = self.bridge_layer1(item_embedding,user_behavior_embedding)
            post_embedding_bridge = self.bridge_layer2(item_embedding, post_embedding)
        else:
            user_behavior_bridge, h_0 = self.bridge_layer1(user_behavior_embedding, None)
            post_embedding_bridge, h_1 =self.bridge_layer2(post_embedding, None)

        browse_atten1 = self.attn1(item_embedding, user_behavior_bridge, browse_mask)
        browse_atten2 = self.attn2(item_embedding, post_embedding_bridge, post_mask)

        concat_feature = torch.cat([item_embedding, browse_atten1, browse_atten2], dim=-1)
        
        # fully-connected layers# output
        output = self.fc_layer(concat_feature)

        return output.squeeze(1).squeeze(1),mi_loss
