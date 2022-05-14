#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:27:26 2020

@author: esat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


import os
import sys
from collections import OrderedDict

from .SlowFast.slowfast_connector import slowfast_50


class rgb_slowfast32f_50(nn.Module):
    def __init__(self, num_classes , length, modelPath='pretrained/SLOWFAST_8x8_R50_torch.pth'):
        super(rgb_slowfast32f_50, self).__init__()
        self.model = slowfast_50(modelPath)
        self.num_classes=num_classes
        self.model.head.dropout = nn.Dropout(0.8)
        self.fc_action = nn.Linear(2304, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.model.head.projection = self.fc_action
    def forward(self, x):
        fast_input = x[:, :, ::1, :, :]
        slow_input = x[:, :, ::4, :, :]
        x = self.model.forward([slow_input, fast_input])
        x = x.view(-1, self.num_classes)
        #x = self.model.forward([fast_input, slow_input])
        return x
