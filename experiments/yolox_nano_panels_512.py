#!/usr/bin/env python3

from yolox_nano_panels import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.input_size = (512, 512)
        self.test_size = (512, 512)
        self.random_size = (16, 16)
        self.exp_name = "yolox_nano_panels_512"
