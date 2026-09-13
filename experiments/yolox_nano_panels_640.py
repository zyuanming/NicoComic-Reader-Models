#!/usr/bin/env python3

from yolox_nano_panels import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (16, 20)
        self.exp_name = "yolox_nano_panels_640"
