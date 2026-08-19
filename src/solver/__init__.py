"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Dict

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .tue_solver import TUESolver
from .calibration_solver import CalibrationSolver

TASKS :Dict[str, BaseSolver] = {
    'classification': ClasSolver,
    'detection': DetSolver,
    'tu_estimation': TUESolver,
    'tue_calibration': CalibrationSolver,
}