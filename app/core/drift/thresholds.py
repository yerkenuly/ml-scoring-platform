from dataclasses import dataclass

from app.config import settings


@dataclass
class DriftThresholdConfig:
    psi_threshold: float = settings.psi_threshold
    ks_p_value_threshold: float = settings.ks_threshold
    performance_degradation_threshold: float = settings.performance_degradation_threshold
    label_psi_threshold: float = 0.1
    score_psi_threshold: float = 0.15
    missing_rate_delta_threshold: float = 0.05


DEFAULT_THRESHOLDS = DriftThresholdConfig()
