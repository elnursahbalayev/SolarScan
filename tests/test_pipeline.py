from pathlib import Path

from PIL import Image

from solarscan.pipeline import run_pipeline
from solarscan.report.pdf import write_pdf_report
from solarscan.schemas import InspectionReport


def _make_image(path: Path, color: int) -> None:
    Image.new("L", (40, 40), color=color).save(path)


def test_pipeline_flags_hot_region(tmp_path):
    img = tmp_path / "hot.png"
    _make_image(img, color=255)  # very bright -> hot-spot in the stub
    report = run_pipeline(img)
    assert isinstance(report, InspectionReport)
    assert report.summary.n_modules_inspected > 1  # detector tiles the frame
    assert report.summary.n_faults == report.summary.n_modules_inspected
    assert all(f.fault_class.value == "Hot-Spot" for f in report.faults)


def test_pipeline_clears_cool_region(tmp_path):
    img = tmp_path / "cool.png"
    _make_image(img, color=10)  # dark -> No-Anomaly -> filtered out
    report = run_pipeline(img)
    assert report.summary.n_faults == 0


def test_pdf_is_written(tmp_path):
    img = tmp_path / "hot.png"
    _make_image(img, color=255)
    report = run_pipeline(img)
    out = write_pdf_report(report, tmp_path / "report.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_detect_only_skips_classification_but_georeferences(tmp_path):
    from solarscan.geo import FarmLayout

    img = tmp_path / "hot.png"
    _make_image(img, color=255)  # would all be "faults" if classified
    report = run_pipeline(img, farm=FarmLayout.sample(), detect_only=True)

    # No fault claims, but every detected module is located + a PDF still renders.
    assert report.summary.n_faults == 0
    assert report.summary.estimated_total_yield_loss_kwh is None
    assert report.summary.n_modules_inspected == len(report.detections) > 0
    assert all(d.location is not None and d.module_id for d in report.detections)
    assert any("Detection-only" in n for n in report.notes)
    out = write_pdf_report(report, tmp_path / "detonly.pdf")
    assert out.exists() and out.stat().st_size > 0
