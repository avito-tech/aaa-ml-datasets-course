import os
from typing import Dict, Optional

import torch
from torch import nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet101,
)

from .settings import NUM_CLASSES


class BaseModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.feature_extractor = None
        self.clf = None
        self.num_classes = num_classes

    def get_model_saving_params(self):
        model_params = {'num_classes': self.num_classes}
        return model_params

    def save_pretrained(self, model_path: str, meta_info: Optional[Dict] = None):
        self.cpu()
        meta_info = meta_info or {}
        meta_info['model_state'] = self.feature_extractor.state_dict()
        meta_info['model_params'] = self.get_model_saving_params()

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(meta_info, model_path)

    @classmethod
    def from_pretrained(cls, model_path):
        model_state = torch.load(model_path)
        instance = cls(**model_state.pop('model_params'))
        instance.feature_extractor.load_state_dict(model_state.pop('model_state'))

        instance.eval()

        return instance, model_state


class RoomModel(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__(num_classes)
        self.feature_extractor = resnet18(pretrained=True)
        # без послденего слоя
        # self.feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
        self.feature_extractor.fc = nn.Linear(
            self.feature_extractor.fc.in_features, out_features=num_classes
        )

    def forward(self, batch):
        out = self.feature_extractor(batch)

        return out
