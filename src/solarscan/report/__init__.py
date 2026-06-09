"""Report stage: yield-loss estimation, severity ranking, and PDF generation."""

from solarscan.report.pdf import write_pdf_report
from solarscan.report.yield_loss import estimate_module_loss_kwh

__all__ = ["estimate_module_loss_kwh", "write_pdf_report"]
