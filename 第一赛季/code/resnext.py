from __future__ import print_function, division, absolute_import
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from resnext_features import resnext101_32x4d_features
from resnext_features import resnext101_64x4d_features

__all__ = ['ResNeXt101_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d','single_task', 'SingleTask']

pretrained_settings = {
    'resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'resnext101_64x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_32x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_32x4d'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def resnext101_64x4d(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_64x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

#add something
class SingleTask(nn.Module):
    def __init__(self, num_classes):
        super(SingleTask, self).__init__()
        #self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        #self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        #self.last_linear = nn.Linear(2048, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    

def load_pre_model_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            print('true')
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def resnex64(num_classes, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = SingleTask(num_classes)
        # model.load_state_dict(model_zoo.load_url(settings['url']))
        load_pre_model_dict(model, model_zoo.load_url(settings['url']))
        
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = SingleTask(num_classes)
    return model