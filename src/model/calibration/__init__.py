# Package initialization
# Import main classes for easy access

"""
Calibration module for probabilistic predictions.

Provides temperature scaling and kernel-based conditional CDF calibration
for improving probabilistic model reliability.

Main classes:
- TemperatureCalibrator: Parametric calibration via temperature scaling
- KCCDCalibrator: Non-parametric calibration via Nadaraya-Watson smoothing
- BetaAdapter, GaussianAdapter: Distribution-specific operations
"""

from .base import Calibrator, DistributionAdapter
from .distributions import BetaAdapter, GaussianAdapter
from .intervals import (
    beta_credible_interval,
    beta_pit,
    interval_report,
    format_interval_report,
)
# from .temperature import TemperatureCalibrator
# from .kccd import KCCDCalibrator
# from .metrics import (
#     pit_values,
#     pit_uniformity_test, 
#     coverage_probability,
#     interval_score,
#     crps_empirical,
#     reliability_diagram
# )

__all__ = [
    # Base classes
    'Calibrator',
    'DistributionAdapter',
    
    # Distribution adapters
    'BetaAdapter', 
    'GaussianAdapter',

    # Credible intervals / coverage of the Beta QC head
    'beta_credible_interval',
    'beta_pit',
    'interval_report',
    'format_interval_report',
    
    # Calibrators
    'TemperatureCalibrator',
    'KCCDCalibrator',
    
    # Metrics
    'pit_values',
    'pit_uniformity_test',
    'coverage_probability', 
    'interval_score',
    'crps_empirical',
    'reliability_diagram'
]