# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

from typing import Dict

import numpy as np

import pyvina_core


class SamplerBase(object):
    """
    """

    def __init__(
        #####
        self,
        core_sampler_base: pyvina_core.samplers.CoreSamplerBase,
    ) -> None:
        """
        """
        super().__init__()
        self.CORE_SAMPLER_BASE = core_sampler_base

    @property
    def has_core_sampler_base(self, ) -> bool:
        """
        """
        return (
            #####
            self.CORE_SAMPLER_BASE
            is not None
        )

    def sample(
        self,
        corner_min: np.ndarray,
        corner_max: np.ndarray,
        num_repeats_monte_carlo: int,
        num_steps_each_monte_carlo: int,
    ) -> Dict:
        """
        """
        assert self.has_core_sampler_base
        tuple_the_sampled = self.CORE_SAMPLER_BASE.Sample(
            #####
            corner_min,
            corner_max,
            num_repeats_monte_carlo,
            num_steps_each_monte_carlo,
        )
        return {
            "poses": [
                #####
                x[0]
                for x in tuple_the_sampled
            ],
            "energies": [
                #####
                x[1]
                for x in tuple_the_sampled
            ],
            "tuple_the_sampled": tuple_the_sampled,
        }
