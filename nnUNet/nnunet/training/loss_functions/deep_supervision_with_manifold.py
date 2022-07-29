#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import torch
from torch import nn


class MultipleOutputLossMixum(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLossMixum, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, pred, y, y_a=None, y_b=None, lam=1, idx=None):
        if y_a is not None and y_b is not None:
            assert isinstance(pred, (tuple, list)), "x must be either tuple or list"
            assert isinstance(y, (tuple, list)), "y must be either tuple or list"
            if self.weight_factors is None:
                weights = [1] * len(pred)
            else:
                weights = self.weight_factors

            if idx == 0:
                l = weights[0] * (lam * self.loss(pred[0], y_a[0]) + (1 - lam) * self.loss(pred[0], y_b[0]))
            else:
                l = weights[0] * self.loss(pred[0], y[0])

            for i in range(1, len(pred)):
                if weights[i] != 0:
                    if idx == i:
                        l += weights[i] * (lam * self.loss(pred[i], y_a[i]) + (1 - lam) * self.loss(pred[i], y_b[i]))
                    else:
                        l += weights[i] * self.loss(pred[i], y[i])
            return l
        else:
            if self.weight_factors is None:
                weights = [1] * len(pred)
            else:
                weights = self.weight_factors

            l = weights[0] * self.loss(pred[0], y[0])
            for i in range(1, len(pred)):
                if weights[i] != 0:
                    l += weights[i] * self.loss(pred[i], y[i])
            return l
