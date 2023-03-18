from ..utils import SOLVER_REGISTRY
from .image_solver import ImageSolver
from .feat_solver import FeatSolver
from .infer_solver import InferSolver
from .valid_solver import ValidSolver
from .ensemble_solver import EnsembleSolver
from .ensemble2_solver import AuEnsembleSolver


def build_solver(modality):
    return SOLVER_REGISTRY.build(modality)