
# model.py
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class TransformerForNextCond(nn.Module):
    def __init__(self, base_model_name, num_labels, dropout=0.2, pooling_strategy='mean'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        self.pooling_strategy = pooling_strategy
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = out.last_hidden_state
        if self.pooling_strategy == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            raise ValueError("Invalid pooling strategy")
        return self.classifier(pooled)