# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

from typing import Dict

import pyvina_core


class EvaluatorBase(object):
    """
    """

    def __init__(
        #####
        self,
        core_evaluator_base: pyvina_core.evaluators.CoreEvaluatorBase,
    ) -> None:
        """
        """
        super().__init__()
        self.CORE_EVALUATOR_BASE = core_evaluator_base

    @property
    def has_core_evaluator_base(self, ) -> bool:
        """
        """
        return (
            #####
            self.CORE_EVALUATOR_BASE
            is not None
        )

    def _evaluate_old1(
        #####
        self,
        conformation,
        calc_gradient=True,
    ) -> Dict:
        assert self.has_core_evaluator_base
        result_evaluation = self.CORE_EVALUATOR_BASE.evaluate(conformation)
        return {
            "energy": result_evaluation[0],
            "gradient": result_evaluation[1],
            "is_valid_conformation": result_evaluation[2],
        }

    evaluate = _evaluate_old1
