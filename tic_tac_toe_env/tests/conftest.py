# SPDX-License-Identifier: BSD-3-Clause

"""Make the package root importable as bare `models` / `server...`, matching
the import style used by train_q_learning.py and the server modules
themselves, regardless of which directory pytest is invoked from."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
