# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

from typing import Dict, Sequence

import pyvina_core


class MinimizerBase(object):
    """
    """

    def __init__(
        #####
        self,
        core_minimizer_base: pyvina_core.minimizers.CoreMinimizerBase,
    ) -> None:
        """
        """
        super().__init__()
        self.CORE_MINIMIZER_BASE = core_minimizer_base

    @property
    def has_core_minimizer_base(self, ) -> bool:
        """
        """
        return (
            #####
            self.CORE_MINIMIZER_BASE
            is not None
        )

    def minimize(
        #####
        self,
        pose: Sequence[float],
    ) -> Dict:
        """
        """
        assert self.has_core_minimizer_base
        # print(" :: call self.CORE_MINIMIZER_BASE.Minimize(pose) :: ")
        tuple_the_minimized = self.CORE_MINIMIZER_BASE.Minimize(pose)
        return {
            "pose": tuple_the_minimized[0],
            "energy": tuple_the_minimized[1],
            "tuple_the_minimized": tuple_the_minimized,
        }
