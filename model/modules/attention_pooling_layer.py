import torch.nn as nn
import torch

from model.modules.fullyconnected_layer import FullyConnectedLayer

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4 * embedding_dim,
                                       hidden_unit=hidden_unit,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='dice',
                                       dice_dim=3)
        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    def forward(self, target_item_embedding, user_behavior_embedding):
        # target_item           : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior_embedding.size(1)
        processed_feat = target_item_embedding.expand(-1, user_behavior_len, -1)
        attention_input = torch.cat([processed_feat, user_behavior_embedding, processed_feat - user_behavior_embedding,
                                     processed_feat * user_behavior_embedding],
                                    dim=-1)

        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output)  # [B, T, 1]

        return attention_score


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], bias=[True, True], embedding_dim=embedding_dim,
                                             batch_norm=False)

    def forward(self, target_item_embedding, user_behavior_embedding, mask=None):
        # target_item           : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size

        attention_score = self.local_att(target_item_embedding, user_behavior_embedding)
        attention_score = torch.transpose(attention_score, 1, 2)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))

        # output
        output = torch.matmul(attention_score, user_behavior_embedding)
        return output