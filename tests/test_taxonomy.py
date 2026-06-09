from solarscan.taxonomy import (
    ALL_CLASSES,
    FaultClass,
    Severity,
    default_severity,
    is_anomaly,
    yield_loss_fraction,
)


def test_twelve_classes():
    assert len(ALL_CLASSES) == 12
    assert FaultClass.NO_ANOMALY in ALL_CLASSES


def test_no_anomaly_is_not_anomaly():
    assert is_anomaly(FaultClass.NO_ANOMALY) is False
    assert is_anomaly(FaultClass.HOT_SPOT) is True


def test_offline_module_is_total_loss_and_critical():
    assert yield_loss_fraction(FaultClass.OFFLINE_MODULE) == 1.0
    assert default_severity(FaultClass.OFFLINE_MODULE) is Severity.CRITICAL


def test_every_class_has_a_profile():
    for c in ALL_CLASSES:
        assert 0.0 <= yield_loss_fraction(c) <= 1.0
        assert isinstance(default_severity(c), Severity)
