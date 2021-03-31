import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class SLUModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config, pretrained_model, pretrained_dir):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model, cache_dir=pretrained_dir)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 23)

        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values)
        # print(outputs[0].shape)
        output = torch.mean(outputs[0], dim=1)
        logits = self.classifier(self.dropout(output))
        # print(logits.shape)
        return logits
