import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_class,feature_dim, **kwargs):
        super(Classifier, self).__init__()
        self.num_class = num_class

        self.fc = nn.Linear(feature_dim, num_class, bias=True)

    def forward(self, pooled_features):

        # classification
        source_clf = self.fc(pooled_features)

        return source_clf
