# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import os
import subprocess
from typing import List


class WrapperObabel(object):
    """
    """

    @staticmethod
    def checkonly_dotformat_in_path(
        #####
        a_path: str,
        dotformat: str,
        should_casefold: bool = True,
    ) -> None:
        """
        """
        assert dotformat.startswith(".")

        a_abspath = os.path.abspath(a_path)

        if should_casefold:
            assert a_abspath.casefold().endswith(
                #####
                dotformat.casefold()
            )
        else:
            assert a_abspath.endswith(dotformat)

    def __init__(self, ) -> None:
        """
        """
        super().__init__()

    def convert_pdb2pdbqt_receptor(
        self, path_pdb_receptor: str, path2save_pdbqt_receptor: str,
    ) -> None:
        """
        """
        self.checkonly_dotformat_in_path(path_pdb_receptor, ".pdb")
        self.checkonly_dotformat_in_path(path2save_pdbqt_receptor, ".pdbqt")

        return self.convert_by_path(
            path_input=path_pdb_receptor,
            path_output=path2save_pdbqt_receptor,
            cmdarg_write_options="-xr",
        )

    def convert_by_path(
        ##### TODO change args naming
        self,
        path_input: str,
        path_output: str,
        cmdarg_read_options: str = None,
        cmdarg_write_options: str = None,
    ) -> None:
        """
        """
        abspath_input = os.path.abspath(path_input)
        abspath_output = os.path.abspath(path_output)

        abspath_cwd = os.path.dirname(abspath_output)
        assert os.path.exists(abspath_cwd)

        cmdargs_suffix = []
        for i_cmdarg, i_prefix_required in [
            (cmdarg_read_options, "-a"),
            (cmdarg_write_options, "-x"),
        ]:
            if i_cmdarg is not None:
                assert i_cmdarg.startswith(i_prefix_required)
                cmdargs_suffix.append(i_cmdarg)

        subprocess.check_call(
            [
                #####
                # "conda",
                # "run",
                # "-n",
                # "env_for_obabel",
                "obabel",
                abspath_input,
                "-O",
                abspath_output,
                *(cmdargs_suffix),
            ],
            cwd=abspath_cwd,
        )
