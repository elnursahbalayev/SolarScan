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
