import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models

class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model_name, transformer_model_name, max_seq_length):
        super(ImageCaptioningModel, self)._init_()
        self.cnn = getattr(models, cnn_model_name)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove classification head

        self.transformer = BertModel.from_pretrained(transformer_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(transformer_model_name)

        # Additional linear layers to map CNN features to the transformer's embedding space
        self.fc_cnn = nn.Linear(self.cnn[-1].in_features, self.transformer.config.hidden_size)
        self.fc_transformer = nn.Linear(self.transformer.config.hidden_size, max_seq_length)
        self.max_seq_length = max_seq_length

    def forward(self, images, captions=None):
        cnn_features = self.cnn(images).flatten(1)
        cnn_features = self.fc_cnn(cnn_features)

        if captions is not None:
            inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_length)
            transformer_outputs = self.transformer(**inputs)
            outputs = self.fc_transformer(transformer_outputs.last_hidden_state + cnn_features.unsqueeze(1))
        else:
            # For inference, captions are not available
            outputs = self.fc_transformer(cnn_features.unsqueeze(1))

        return outputs