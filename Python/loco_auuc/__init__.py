"""LOCO-AUUC uplift monitoring under ref/live distribution shift."""

from .metrics import auuc, evaluate_window, qini_coefficient
from .monitor import run_uplift_monitor
from .subset_localization import run_uplift_subset_localization, business_rules_from_localization, diagnose_shift
from .loco import loco_auuc_monitor
from .neural_uplift import make_dragonnet, make_tarnet, NeuralUpliftLearner

__all__ = [
    "auuc",
    "evaluate_window",
    "qini_coefficient",
    "run_uplift_monitor",
    "diagnose_shift",
    "loco_auuc_monitor",
    "NeuralUpliftLearner",
    "make_dragonnet",
    "make_tarnet",
]
