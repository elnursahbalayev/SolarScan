"""Transparent yield-loss estimation.

The estimate is intentionally simple and its assumptions are surfaced in the
report: lost energy = module rated power x fault loss-fraction x sun-hours/day.
This is an engineering estimate to prioritise O&M effort, not a metered value.
"""

from __future__ import annotations

from solarscan.taxonomy import FaultClass, yield_loss_fraction

# Defaults; overridable per site via config.
DEFAULT_MODULE_RATED_KW = 0.4  # 400 Wp module
DEFAULT_PEAK_SUN_HOURS = 4.5  # daily equivalent full-sun hours


def estimate_module_loss_kwh(
    fault: FaultClass,
    module_rated_kw: float = DEFAULT_MODULE_RATED_KW,
    peak_sun_hours: float = DEFAULT_PEAK_SUN_HOURS,
) -> float:
    """Estimated daily energy lost (kWh) for one module exhibiting ``fault``."""
    return module_rated_kw * yield_loss_fraction(fault) * peak_sun_hours
