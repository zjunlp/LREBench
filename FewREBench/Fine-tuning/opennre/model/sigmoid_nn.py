import torch
from torch import nn, optim
from .base_model import SentenceRE

class SigmoidNN(SentenceRE):
    """
    Sigmoid (binary) classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        return logits

    def logit_to_score(self, logits):
        return torch.sigmoid(logits)
