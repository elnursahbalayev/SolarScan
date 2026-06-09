"""IEC-aligned PV fault taxonomy.

Classes mirror the InfraredSolarModules dataset (Raptor Maps, ICLR 2020):
11 anomaly classes + No-Anomaly. Each anomaly carries a default severity and a
yield-loss factor used by the report stage. The yield-loss factors are
transparent, literature-informed *estimates* (fraction of the affected module's
output lost while the fault persists) — they are assumptions, not measurements,
and are stated as such in the generated report.
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Severity ordering for ranking.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.NONE: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class FaultClass(str, Enum):
    NO_ANOMALY = "No-Anomaly"
    CELL = "Cell"
    CELL_MULTI = "Cell-Multi"
    CRACKING = "Cracking"
    HOT_SPOT = "Hot-Spot"
    HOT_SPOT_MULTI = "Hot-Spot-Multi"
    SHADOWING = "Shadowing"
    DIODE = "Diode"
    DIODE_MULTI = "Diode-Multi"
    VEGETATION = "Vegetation"
    SOILING = "Soiling"
    OFFLINE_MODULE = "Offline-Module"


# Ordered list matching the dataset's label indices (alphabetical-ish per source).
ALL_CLASSES: list[FaultClass] = list(FaultClass)


# (default severity, estimated yield-loss fraction of the affected module).
# Sources to firm up in the model card; placeholders are deliberately conservative.
FAULT_PROFILE: dict[FaultClass, tuple[Severity, float]] = {
    FaultClass.NO_ANOMALY: (Severity.NONE, 0.00),
    FaultClass.CELL: (Severity.LOW, 0.03),
    FaultClass.CELL_MULTI: (Severity.MEDIUM, 0.08),
    FaultClass.CRACKING: (Severity.MEDIUM, 0.05),
    FaultClass.HOT_SPOT: (Severity.HIGH, 0.10),
    FaultClass.HOT_SPOT_MULTI: (Severity.HIGH, 0.20),
    FaultClass.SHADOWING: (Severity.LOW, 0.15),
    FaultClass.DIODE: (Severity.MEDIUM, 0.33),  # one bypass diode ~ 1/3 of module
    FaultClass.DIODE_MULTI: (Severity.HIGH, 0.66),
    FaultClass.VEGETATION: (Severity.LOW, 0.10),
    FaultClass.SOILING: (Severity.LOW, 0.05),
    FaultClass.OFFLINE_MODULE: (Severity.CRITICAL, 1.00),
}


def default_severity(fault: FaultClass) -> Severity:
    return FAULT_PROFILE[fault][0]


def yield_loss_fraction(fault: FaultClass) -> float:
    """Estimated fraction of the affected module's output lost while the fault persists."""
    return FAULT_PROFILE[fault][1]


def is_anomaly(fault: FaultClass) -> bool:
    return fault is not FaultClass.NO_ANOMALY
