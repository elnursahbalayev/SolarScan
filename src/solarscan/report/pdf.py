"""Generate the severity-ranked PDF inspection report from an InspectionReport."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from solarscan.schemas import InspectionReport
from solarscan.taxonomy import SEVERITY_RANK, Severity


def write_pdf_report(report: InspectionReport, out_path: str | Path) -> Path:
    """Render ``report`` to a PDF at ``out_path``; returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title="SolarScan Inspection Report")
    flow = []

    flow.append(Paragraph("SolarScan — PV Inspection Report", styles["Title"]))
    flow.append(
        Paragraph(
            f"Source: {report.source} &nbsp;|&nbsp; "
            f"Generated: {report.generated_at:%Y-%m-%d %H:%M UTC} &nbsp;|&nbsp; "
            f"Model: {report.model_version}",
            styles["Normal"],
        )
    )
    flow.append(Spacer(1, 0.5 * cm))

    s = report.summary
    flow.append(Paragraph("Summary", styles["Heading2"]))
    summary_rows = [
        ["Modules inspected", str(s.n_modules_inspected)],
        ["Faults found", str(s.n_faults)],
    ]
    if s.estimated_total_yield_loss_kwh is not None:
        summary_rows.append(
            ["Est. daily yield loss", f"{s.estimated_total_yield_loss_kwh:.2f} kWh/day"]
        )
    summary_tbl = Table(summary_rows, colWidths=[6 * cm, 6 * cm])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
            ]
        )
    )
    flow.append(summary_tbl)
    flow.append(Spacer(1, 0.5 * cm))

    flow.append(Paragraph("Faults (severity-ranked)", styles["Heading2"]))
    ranked = sorted(
        report.faults,
        key=lambda f: (SEVERITY_RANK[f.severity], f.confidence),
        reverse=True,
    )
    header = ["#", "Fault", "Severity", "Conf.", "Yield loss", "Location"]
    rows = [header]
    for i, f in enumerate(ranked, 1):
        loc = f"{f.location.lat:.5f}, {f.location.lon:.5f}" if f.location else "—"
        rows.append(
            [
                str(i),
                f.fault_class.value,
                f.severity.value,
                f"{f.confidence:.2f}",
                f"{f.estimated_yield_loss_fraction * 100:.0f}%",
                loc,
            ]
        )
    faults_tbl = Table(rows, repeatRows=1)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]
    for ridx, f in enumerate(ranked, start=1):
        if f.severity in (Severity.HIGH, Severity.CRITICAL):
            style.append(("TEXTCOLOR", (2, ridx), (2, ridx), colors.red))
    faults_tbl.setStyle(TableStyle(style))
    flow.append(faults_tbl)

    if report.notes:
        flow.append(Spacer(1, 0.5 * cm))
        flow.append(Paragraph("Notes & assumptions", styles["Heading2"]))
        for note in report.notes:
            flow.append(Paragraph(f"• {note}", styles["Normal"]))

    doc.build(flow)
    return out_path
