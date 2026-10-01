#!/usr/bin/env python3
"""Genera Informe_Lab6.pdf (máx. ~4 páginas) a partir de outputs_lab6/."""

from pathlib import Path
import json
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
    )
    from reportlab.lib import colors
except ImportError:
    import subprocess, sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-q", "reportlab"])
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
    )
    from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_lab6"
FIG = ROOT / "figures"


def main():
    val = pd.read_csv(OUT / "validation_runs.csv")
    test = pd.read_csv(OUT / "test_results.csv")
    red = pd.read_csv(OUT / "reduced_10pct_results.csv")
    final = pd.read_csv(OUT / "final_comparison.csv")
    meta = json.loads((OUT / "run_meta.json").read_text())

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Tiny", fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Body2", fontSize=9, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2c", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold"))

    doc = SimpleDocTemplate(
        str(ROOT / "Informe_Lab6.pdf"),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    story = []
    story.append(Paragraph("CC3092 — Laboratorio 6: Transfer Learning y Fine-Tuning", styles["Title"]))
    story.append(Paragraph("CIFAR-10 · CNN desde cero · VGG-16 feature extractor · VGG-16 fine-tuning (112×112)", styles["Body2"]))
    story.append(Paragraph(f"Hardware de entrenamiento: {meta.get('device_reported','GPU CUDA')}. Semilla {meta['seed']}.", styles["Tiny"]))

    story.append(Paragraph("1. Investigación: transfer learning en PyTorch", styles["H2c"]))
    story.append(
        Paragraph(
            "<b>vgg16 / VGG16_Weights</b> cargan la arquitectura y pesos ImageNet; "
            "<b>weights.transforms()</b> define el preprocesado oficial (224×224). "
            "<b>features / avgpool / classifier</b> separan backbone, pooling adaptativo y cabeza MLP. "
            "<b>requires_grad</b> congela o descongela bloques. "
            "<b>train/eval</b> y <b>no_grad</b> controlan Dropout/BN e inferencia. "
            "<b>param_groups</b> permiten LR distinto en backbone y clasificador. "
            "<b>fvcore</b> estima FLOPs; <b>max_memory_allocated / synchronize</b> miden memoria y tiempo en GPU. "
            "Un MAC ≈ 2 FLOPs. Parámetros entrenables ⊆ totales.",
            styles["Body2"],
        )
    )
    story.append(
        Paragraph(
            "Se usa <b>112×112</b> (no 224) por cómputo: ≈12.25× píxeles vs 32×32 (224 sería 49×), "
            "misma resolución en FE y FT.",
            styles["Body2"],
        )
    )

    story.append(Paragraph("2. Resultados de las 9 iteraciones (validación)", styles["H2c"]))
    cols = ["experiment", "unfrozen_blocks", "best_epoch", "best_val_f1", "best_val_accuracy", "trainable_params", "total_training_time_s"]
    cols = [c for c in cols if c in val.columns]
    data = [cols] + [
        [
            str(r[c]) if not isinstance(r[c], float) else (f"{r[c]:.4f}" if "f1" in c or "acc" in c else f"{r[c]:.1f}" if "time" in c else str(int(r[c]) if c.endswith("params") or c.endswith("epoch") else r[c]))
            for c in cols
        ]
        for _, r in val.sort_values(["model_family", "best_val_f1"], ascending=[True, False]).iterrows()
    ]
    # simplify
    data = [cols]
    for _, r in val.sort_values(["model_family", "best_val_f1"], ascending=[True, False]).iterrows():
        row = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                row.append("-")
            elif c in ("best_val_f1", "best_val_accuracy"):
                row.append(f"{float(v):.4f}")
            elif c == "total_training_time_s":
                row.append(f"{float(v):.0f}")
            elif c in ("trainable_params", "best_epoch"):
                row.append(str(int(v)))
            else:
                row.append(str(v))
        data.append(row)
    t = Table(data, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t)

    story.append(Paragraph("3. Comparación de rendimiento y recursos (mejores modelos)", styles["H2c"]))
    fcols = [
        "family",
        "test_f1",
        "test_accuracy",
        "f1_10pct",
        "total_params",
        "trainable_params",
        "flops_per_image",
        "latency_ms",
        "peak_gpu_memory_mb",
    ]
    fdata = [fcols]
    for _, r in final.iterrows():
        fdata.append(
            [
                r["family"],
                f"{r['test_f1']:.4f}",
                f"{r['test_accuracy']:.4f}",
                f"{r['f1_10pct']:.4f}",
                f"{int(r['total_params']):,}",
                f"{int(r['trainable_params']):,}",
                f"{r['flops_per_image']/1e9:.3f}G",
                f"{r['latency_ms']:.2f}",
                f"{r['peak_gpu_memory_mb']:.0f}",
            ]
        )
    t2 = Table(fdata, repeatRows=1)
    t2.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t2)

    for img_name, w in [("val_acc_vs_time.png", 5.8), ("f1_vs_flops.png", 4.8)]:
        p = FIG / img_name
        if p.exists():
            story.append(Spacer(1, 6))
            story.append(Image(str(p), width=w * inch, height=w * inch * 0.62))

    story.append(Paragraph("4. Análisis y conclusiones", styles["H2c"]))
    winner = final.sort_values("test_f1", ascending=False).iloc[0]
    cnn = final[final.family == "CNN"].iloc[0]
    fe = final[final.family == "VGG-FE"].iloc[0]
    ft = final[final.family == "VGG-FT"].iloc[0]
    story.append(
        Paragraph(
            f"El mejor desempeño en test es <b>{winner['family']}</b> "
            f"(F1={winner['test_f1']:.4f}, acc={winner['test_accuracy']:.4f}). "
            f"La CNN alcanza F1={cnn['test_f1']:.4f} con mucho menos cómputo "
            f"({cnn['flops_per_image']/1e6:.1f} MFLOPs/imagen). "
            f"FE ({fe['test_f1']:.4f}) mejora a la CNN entrenando solo el clasificador; "
            f"FT aporta +{(ft['test_f1']-fe['test_f1']):.4f} F1 descongelando bloques superiores "
            f"con LR bajo en el backbone, sin indicios claros de catastrophic forgetting.",
            styles["Body2"],
        )
    )
    story.append(
        Paragraph(
            "Con el 10 % de datos, la CNN es la más afectada; FE/FT degradan menos gracias a ImageNet. "
            "Los errores recurrentes (cat↔dog, auto↔truck, bird↔airplane) aparecen en los tres modelos. "
            "Despliegue: GPU→VGG-FT; móvil→CNN o MobileNetV3/EfficientNet-B0.",
            styles["Body2"],
        )
    )
    story.append(Paragraph("Repositorio: ver notebook <b>Laboratorio6_Transfer_Learning.ipynb</b> y carpeta <b>outputs_lab6/</b>.", styles["Tiny"]))

    doc.build(story)
    print("PDF escrito:", ROOT / "Informe_Lab6.pdf")


if __name__ == "__main__":
    main()
