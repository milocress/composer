# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""MUP optimizer class and wrappers."""

from __future__ import annotations

import logging
from typing import Optional

from mup import MuAdam, MuSGD, MuReadout, set_base_shapes
from torchvision.models.resnet import ResNet, Bottleneck

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import ensure_tuple

log = logging.getLogger(__name__)

__all__ = ['MUP']

def resnet_surgery(model: ResNet) -> ResNet:
    model.fc = MuReadout(model.fc.in_features, model.fc.out_features, readout_zero_init=True)
    return model

class MUP(Algorithm):
    """Mu Transfer algorithm (TODO: expand stub)

    Args: 
        optimizer_family (str): Optimizer family
        model_family (str): Model family
    """
    def __init__(
        self,
        optimizer_family: str,
        model_family: str,
    ):
        """__init__ is constructed from the same fields as in hparams."""
        self.optimizer_family = optimizer_family
        self.model_family = model_family

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger]) -> Optional[int]:
        assert state.optimizers is not None

        wrapper = None

        if self.optimizer_family == 'adam':
            wrapper = MuAdam
        elif self.optimizer_family == 'sgd':
            wrapper = MuSGD
        else:
            raise ValueError(f'Unknown optimizer family: {self.optimizer_family}')

        if self.model_family == 'resnet':
            model = resnet_surgery(state.model.module)
            resnet_small = resnet_surgery(ResNet(Bottleneck, [2,2,2,2]))
            resnet_delta = resnet_surgery(ResNet(Bottleneck, [3,4,6,3]))
            set_base_shapes(model, resnet_small, delta=resnet_delta)

        else:
            raise ValueError(f'Unknown model family: {self.model_family}')


        def construct_optimizer(base_optimizer):
            def MuOptimizer(params, **kwargs):
                return wrapper(params, impl=base_optimizer, **kwargs)

            return MuOptimizer

        state.optimizers = tuple(
            construct_optimizer(
                base_optimizer=optimizer,
            ) for optimizer in ensure_tuple(state.optimizers))
