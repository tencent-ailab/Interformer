# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import pyvina_core

from pdbqt_ligand.components.minimizers.minimizer_base import MinimizerBase
from pdbqt_ligand.components.samplers.sampler_base import SamplerBase


class SamplerMonteCarlo(SamplerBase):
    """
    """

    def __init__(
        #####
        self,
        a_minimizer: MinimizerBase,
    ) -> None:
        assert a_minimizer.has_core_minimizer_base
        self._CORE_SAMPLER_MONTE_CARLO = pyvina_core.samplers.CoreSamplerMonteCarlo(
            a_minimizer.CORE_MINIMIZER_BASE
        )
        super().__init__(
            #####
            core_sampler_base=self._CORE_SAMPLER_MONTE_CARLO,
        )
