#!/usr/bin/env python3

import os

import torch.nn as nn

from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.max_epoch = 100
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        def initialize(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

            channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=channels, act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=channels, act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head)
        self.model.apply(initialize)
        self.model.head.initialize_biases(1e-2)
        return self.model
