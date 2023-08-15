import torch
import torch.nn as nn
from model.modules.mlp import MLP
from model.modules.fullyconnected_layer import FullyConnectedLayer
from model.modules.attention_pooling_layer import AttentionSequencePoolingLayer
from dataset.loader import load_pretrained_embedding,load_corresponding_iv
from config.config import Iv4RecConfig

embedding_size = Iv4RecConfig.embedding_size

# IV4REC
class IV4Rec(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = Iv4RecConfig.description
        self.device = Iv4RecConfig.device

        self.regularization_weight = []

        item_embedding_matrix, post_embedding_matrix = load_pretrained_embedding()
        self.item_embedding_layer = nn.Embedding.from_pretrained(item_embedding_matrix, freeze=True)
        self.post_embedding_layer = nn.Embedding.from_pretrained(post_embedding_matrix, freeze=True)

        _, _, query_embedding_matrix, cor_query_matrix = load_corresponding_iv()
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_matrix, freeze=True)

        #MLP0
        self.mlp = MLP(Iv4RecConfig.mlp)
        self.add_regularization_weight(self.mlp.parameters(), l2=Iv4RecConfig.l2_lambda)

        #MLP1 and MLP2
        self.alpha = MLP(Iv4RecConfig.alpha)
        self.add_regularization_weight(self.alpha.parameters(), l2=Iv4RecConfig.l2_lambda)
        self.beta = MLP(Iv4RecConfig.beta)
        self.add_regularization_weight(self.beta.parameters(), l2=Iv4RecConfig.l2_lambda)

        # attention层
        self.attn1 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.attn2 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)

        self.embedder = FullyConnectedLayer(input_size=768, hidden_unit=[embedding_size], bias=[False],
                                            batch_norm=False, sigmoid=False, activation='dice', dice_dim=1)

        #全连接层
        self.fc_layer = FullyConnectedLayer(input_size=3 * embedding_size,
                                            hidden_unit=[200, 80, 1],
                                            bias=[True, True, False],
                                            batch_norm=False,
                                            sigmoid=True,
                                            activation='dice',
                                            dice_dim=3)

        self.prob_sigmoid = nn.Sigmoid()

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

    def iv(self, item, query, query_pinv):
        '''
        input:
            item:<batch,1,embedding_size>
            query:<batch,1,embedding_size,10>
            query_pinv:<batch,1,10,embedding_size>
        output：
            reconstruct_embedding:<batch,1,embedding_size>
        '''
        query_origin = query.flatten(start_dim= 2) # <batch,1,embedding_size*30>

        alpha = self.prob_sigmoid(self.alpha(torch.cat([item, query_origin], dim=-1))) # <batch,1,1>
        beta = self.prob_sigmoid(self.beta(torch.cat([item, query_origin], dim=-1))) # <batch,1,1>

        item = item.unsqueeze(dim=-1) # <batch,1,embedding_size,1>
        t_1 = torch.matmul(query, torch.matmul(query_pinv, item)) #  <batch,1,embedding_size,1>

        q_t = torch.matmul(item, alpha.unsqueeze(dim=-1)) + \
              torch.matmul(t_1, beta.unsqueeze(dim=-1))

        return q_t.squeeze_(dim=-1)  # <batch,1,embedding_size>

    def forward(self, x):
        '''
        input:
            search_log : <batch, query_history>
            behavior_history : <batch, hist_number>
            thread_click_item : <batch, query_history>
            item : <batch>
        '''
        x = [i.to(self.device) for i in x]
        user, behavior_history, item, click_post, query = x

        cor_query_embedding = self.embedder(self.query_embedding_layer(query))
        cor_query_embedding = cor_query_embedding.mean(dim=1, keepdim=True)
        cor_query_embedding = cor_query_embedding.reshape(item.shape[0], 1, embedding_size, 1)
        cor_query_pinv = self.embedder(self.cor_query_pinv(query))
        cor_query_pinv = cor_query_pinv.mean(dim=1, keepdim=True)
        cor_query_pinv = cor_query_pinv.reshape(item.shape[0], 1, 1, embedding_size)
        browse_cor_query_embedding = cor_query_embedding.repeat(1, 20, 1, 1)
        browse_cor_query_pinv = cor_query_pinv.repeat(1, 20, 1, 1)

        item_embedding = self.embedder(self.item_embedding_layer(item))  # <batch, embedding_size>
        item_embedding = self.mlp(item_embedding).reshape(item.shape[0], 1, embedding_size)  # batch, 1, feature
        item_embedding = self.iv(item_embedding, cor_query_embedding, cor_query_pinv)  # batch, 1, feature

        browse_embedding = self.embedder(
            self.item_embedding_layer(behavior_history))  # <batch, hist_number, embedding_size>
        browse_embedding = self.mlp(browse_embedding)  # batch, len, feature
        browse_embedding = self.iv(browse_embedding, browse_cor_query_embedding,
                                   browse_cor_query_pinv)  # batch, len, feature
        browse_mask = torch.where(behavior_history == 0, 1, 0).bool()  # <batch, hist_number>

        thread_embedding = self.embedder(self.post_embedding_layer(click_post))  # <batch, hist_number, embedding_size>
        thread_embedding_mask = torch.where(click_post == 0, 1, 0).bool()  # <batch, query_history>

        # underlying model：attention-concat-fc
        # attention
        thread_attetn = self.attn1(item_embedding, browse_embedding, browse_mask)
        click_atten = self.attn2(item_embedding, thread_embedding, thread_embedding_mask)  # <batch,1,embedding_size>
        concat_feature = torch.cat([item_embedding, thread_attetn, click_atten], dim=-1)  # <batch,1,embedding_size*4>

        # 全连接层
        output = self.fc_layer(concat_feature)  # <batch,1,1>

        return output.squeeze(1).squeeze(1)
