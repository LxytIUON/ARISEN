import torch
import torch.nn as nn
from model.modules.fullyconnected_layer import FullyConnectedLayer
from model.modules.attention_pooling_layer import AttentionSequencePoolingLayer
from config.config import DinConfig
from dataset.loader import load_pretrained_embedding

from model.modules.mlp import MLP

embedding_size = DinConfig.embedding_size

# DIN
class DeepInterestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'din'
        self.device = DinConfig.device

        self.regularization_weight = []

        photo_embedding_matrix,_ = load_pretrained_embedding()

        self.photo_embedding_layer = nn.Embedding.from_pretrained(photo_embedding_matrix, freeze=True)

        self.attn = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        # embedding
        self.embedder = FullyConnectedLayer(input_size=768, hidden_unit=[512,256,128,embedding_size], bias=[False,False,False,False],batch_norm=False, sigmoid=False, activation='relu', dice_dim=4)

        self.fc_layer = FullyConnectedLayer(input_size=2 * embedding_size,hidden_unit=[200, 80, 1],bias=[True, True, False],batch_norm=False,sigmoid=True,activation='prelu',dice_dim=3)


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
   
    def forward(self, x):
        user, user_behavior, target_item = x
        user,user_behavior, target_item = user.to(self.device),user_behavior.to(self.device), target_item.to(self.device)

        user_behavior_embedding = self.embedder(self.photo_embedding_layer(user_behavior))
        browse_mask = torch.where(user_behavior == 0, 1, 0).bool()

        target_item_embedding = self.embedder(self.photo_embedding_layer(target_item))
        target_item_embedding = target_item_embedding.unsqueeze(1)

        # attn
        target_item_embedding = target_item_embedding.reshape(target_item.shape[0], 1, embedding_size)
        browse_atten = self.attn(target_item_embedding,user_behavior_embedding, browse_mask)
        # concat
        concat_feature = torch.cat([target_item_embedding, browse_atten], dim=-1)

        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return output.squeeze(1).squeeze(1)
