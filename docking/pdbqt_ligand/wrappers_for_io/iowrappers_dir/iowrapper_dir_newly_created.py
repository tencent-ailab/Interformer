# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com

"""
"""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Tuple


class IowrapperDirNewlyCreated(object):
    """
    wrapper for a dir that has not been created before
    and will be created in __init__
    """

    def __init__(
        #####
        self,
        path_dir_nonexistent: str = None,
        safe_copy: bool = True,
    ) -> None:
        """
        if [ path_dir_nonexistent ]=[ None ], a new temporary dir will be created;
        if [ path_dir_nonexistent ]=[ X ], [ X ] should be a path that has not been created before (not exists);
        """
        super().__init__()

        #######################################
        ##### 1. abspath for new dir
        #######################################
        abspath_dir_newly_created = None

        if path_dir_nonexistent is None:
            abspath_dir_newly_created = os.path.abspath(
                #####
                tempfile.mkdtemp()
            )
        else:
            assert isinstance(path_dir_nonexistent, str)
            abspath_dir_nonexistent = os.path.abspath(path_dir_nonexistent)

            assert not os.path.exists(abspath_dir_nonexistent)
            Path(abspath_dir_nonexistent).mkdir(
                #####
                parents=True,
                exist_ok=False,
            )

            abspath_dir_newly_created = abspath_dir_nonexistent

        self.ABSPATH_DIR_NEWLY_CREATED = abspath_dir_newly_created

        #######################################
        ##### 2. safe copy
        #######################################
        assert isinstance(safe_copy, bool)
        self.SAFE_COPY = safe_copy

    def copy_files_inward(self, *args, ) -> Tuple[str, ...]:
        """will return (path_cwd, path_file_new_1, path_file_new_2, ...)
        """
        ##### 1. checks
        for i_arg in args:
            assert isinstance(i_arg, str)

        ##### 2. pathes raw
        abspathes_to_copy = [
            #####
            os.path.abspath(x)
            for x in args
        ]

        ##### 3. copy
        abspathes_new = []
        for i_abspath_to_copy in abspathes_to_copy:
            i_abspath_new = os.path.join(
                #####
                self.ABSPATH_DIR_NEWLY_CREATED,
                os.path.basename(i_abspath_to_copy),
            )

            ##### no overwriting
            if self.SAFE_COPY:
                assert not os.path.exists(i_abspath_new)

            shutil.copyfile(
                #####
                i_abspath_to_copy,
                i_abspath_new,
            )
            abspathes_new.append(i_abspath_new)

        return (
            self.ABSPATH_DIR_NEWLY_CREATED,
            *(abspathes_new),
        )
