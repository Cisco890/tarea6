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
        KeepTogether,
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
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
        KeepTogether,
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_lab6"
FIG = ROOT / "figures"

AUTHOR = "Juan Francisco Martínez"
CARNET = "23617"
REPO_URL = "https://github.com/Cisco890/tarea6"


def _table_style(font_size=7):
    return TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def main():
    val = pd.read_csv(OUT / "validation_runs.csv")
    test = pd.read_csv(OUT / "test_results.csv")
    red = pd.read_csv(OUT / "reduced_10pct_results.csv")
    final = pd.read_csv(OUT / "final_comparison.csv")
    meta = json.loads((OUT / "run_meta.json").read_text())
    stats = json.loads((OUT / "channel_stats.json").read_text())

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleC",
            parent=styles["Title"],
            fontSize=14,
            leading=17,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(ParagraphStyle(name="Meta", fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=2))
    styles.add(ParagraphStyle(name="Tiny", fontSize=8, leading=10, spaceAfter=3))
    styles.add(ParagraphStyle(name="Body2", fontSize=9, leading=12, spaceAfter=5))
    styles.add(
        ParagraphStyle(
            name="H2c",
            fontSize=11,
            leading=13,
            spaceBefore=8,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3c",
            fontSize=9.5,
            leading=12,
            spaceBefore=5,
            spaceAfter=2,
            fontName="Helvetica-Bold",
        )
    )

    doc = SimpleDocTemplate(
        str(ROOT / "Informe_Lab6.pdf"),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    story = []

    # Portada / encabezado
    story.append(Paragraph("CC3092 — Deep Learning y Sistemas Inteligentes", styles["Meta"]))
    story.append(Paragraph("Laboratorio 6: Transfer Learning y Fine-Tuning", styles["TitleC"]))
    story.append(
        Paragraph(
            f"<b>{AUTHOR}</b> &nbsp;|&nbsp; Carnet <b>{CARNET}</b>",
            styles["Meta"],
        )
    )
    story.append(
        Paragraph(
            f'Repositorio: <link href="{REPO_URL}">{REPO_URL}</link>',
            styles["Meta"],
        )
    )
    story.append(
        Paragraph(
            "CIFAR-10 · CNN desde cero · VGG-16 feature extractor · VGG-16 fine-tuning (112×112)",
            styles["Tiny"],
        )
    )
    story.append(
        Paragraph(
            f"Hardware: {meta.get('device_reported', 'GPU CUDA')}. Semilla {meta['seed']}. "
            f"Split estratificado 45,000 / 5,000; test oficial de 10,000 imágenes.",
            styles["Tiny"],
        )
    )
    story.append(Spacer(1, 4))

    # 1. Investigación
    story.append(Paragraph("1. Investigación: transfer learning en PyTorch", styles["H2c"]))
    story.append(
        Paragraph(
            "<b>torchvision.models.vgg16</b> y <b>VGG16_Weights.DEFAULT</b> construyen la red e "
            "inicializan pesos de ImageNet. <b>weights.transforms()</b> entrega el pipeline oficial "
            "(resize a 224×224 y normalización ImageNet); aquí se sustituye por un pipeline propio a "
            "<b>112×112</b> por costo computacional, manteniendo la misma resolución en feature "
            "extraction y fine-tuning.",
            styles["Body2"],
        )
    )
    story.append(
        Paragraph(
            "La arquitectura se divide en <b>model.features</b> (bloques convolucionales), "
            "<b>model.avgpool</b> (<i>AdaptiveAvgPool2d</i>, tamaño espacial fijo) y "
            "<b>model.classifier</b> (MLP; la última <i>Linear</i> se reemplaza por 10 salidas). "
            "Con <b>requires_grad</b> se congelan o descongelan bloques. "
            "<b>train()/eval()</b> controlan Dropout y BatchNorm; <b>torch.no_grad()</b> evita "
            "guardar el grafo en inferencia. En fine-tuning se usan <b>param_groups</b> del "
            "optimizador para asignar un learning rate menor al backbone que al clasificador.",
            styles["Body2"],
        )
    )
    story.append(
        Paragraph(
            "Para costo computacional se usa <b>fvcore.FlopCountAnalysis</b> (FLOPs del forward). "
            "En GPU, <b>torch.cuda.max_memory_allocated</b> y <b>synchronize</b> permiten reportar "
            "memoria pico y tiempos fiables. Un MAC ≈ 2 FLOPs; se mantiene la misma convención "
            "entre modelos. Los parámetros entrenables son el subconjunto con "
            "<i>requires_grad=True</i>.",
            styles["Body2"],
        )
    )
    mean, std = stats["cifar_mean"], stats["cifar_std"]
    story.append(
        Paragraph(
            f"En datos: CIFAR-10 está balanceado (5,000 imgs/clase en train). Media/std por canal "
            f"calculadas en train: "
            f"R={mean[0]:.3f}±{std[0]:.3f}, G={mean[1]:.3f}±{std[1]:.3f}, B={mean[2]:.3f}±{std[2]:.3f} "
            f"(ImageNet usa 0.485/0.456/0.406 y 0.229/0.224/0.225). La CNN normaliza con CIFAR; "
            f"VGG con ImageNet. Augmentation solo en train. "
            f"Pasar de 32×32 a 112×112 multiplica píxeles por ≈12.25× (224×224 sería 49×).",
            styles["Body2"],
        )
    )

    # 2. Iteraciones
    story.append(Paragraph("2. Resultados de las 9 iteraciones (validación)", styles["H2c"]))
    story.append(
        Paragraph(
            "Se ejecutaron tres configuraciones por familia. Selección del mejor modelo "
            "<b>únicamente por F1 macro de validación</b> (sin mirar test). Early stopping con "
            "paciencia 3. CNN en 32×32; VGG en 112×112.",
            styles["Body2"],
        )
    )
    cols = [
        "experiment",
        "unfrozen_blocks",
        "best_epoch",
        "best_val_f1",
        "best_val_accuracy",
        "trainable_params",
        "total_training_time_s",
    ]
    cols = [c for c in cols if c in val.columns]
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
    t = Table(data, repeatRows=1, colWidths=[72, 70, 48, 58, 70, 78, 72])
    t.setStyle(_table_style(7))
    story.append(t)
    story.append(
        Paragraph(
            "Mejores por familia: <b>CNN-A</b> (LR=1e-3, dropout=0.3), <b>VGG-FE-B</b> "
            "(LR clasificador=3e-4) y <b>VGG-FT-B</b> (bloques 4+5, LR backbone=5e-6, "
            "clasificador=1e-4). En fine-tuning, descongelar 4+5 superó a solo el bloque 5.",
            styles["Body2"],
        )
    )

    # 3. Comparación
    story.append(Paragraph("3. Comparación de rendimiento y recursos", styles["H2c"]))
    story.append(Paragraph("3.1 Métricas de test y experimento con 10 % de datos", styles["H3c"]))

    # Drop table
    drop_data = [["Modelo", "Test Acc", "Test F1", "F1 10%", "Caída abs.", "Caída rel."]]
    for fam in ["CNN", "VGG-FE", "VGG-FT"]:
        te = test[test.family == fam].iloc[0]
        rd = red[red.family == fam].iloc[0]
        abs_d = float(te["f1"]) - float(rd["f1"])
        rel_d = abs_d / float(te["f1"])
        drop_data.append(
            [
                fam,
                f"{te['accuracy']:.4f}",
                f"{te['f1']:.4f}",
                f"{rd['f1']:.4f}",
                f"{abs_d:.4f}",
                f"{rel_d*100:.1f}%",
            ]
        )
    t_drop = Table(drop_data, repeatRows=1)
    t_drop.setStyle(_table_style(8))
    story.append(t_drop)
    story.append(Spacer(1, 4))

    story.append(Paragraph("3.2 Recursos de los mejores modelos", styles["H3c"]))
    fcols = [
        "family",
        "test_f1",
        "f1_10pct",
        "total_params",
        "trainable_params",
        "flops_per_image",
        "latency_ms",
        "peak_gpu_memory_mb",
    ]
    fdata = [["family", "test_f1", "f1_10%", "params", "trainable", "FLOPs/img", "lat. ms", "VRAM MB"]]
    for _, r in final.iterrows():
        fdata.append(
            [
                r["family"],
                f"{r['test_f1']:.4f}",
                f"{r['f1_10pct']:.4f}",
                f"{int(r['total_params']):,}",
                f"{int(r['trainable_params']):,}",
                f"{r['flops_per_image']/1e9:.3f}G",
                f"{r['latency_ms']:.2f}",
                f"{r['peak_gpu_memory_mb']:.0f}",
            ]
        )
    t2 = Table(fdata, repeatRows=1)
    t2.setStyle(_table_style(7.5))
    story.append(t2)
    story.append(
        Paragraph(
            "La estimación de FLOPs de entrenamiento usa "
            "<i>FLOPs_forward × #imágenes × epochs_hasta_mejor × factor</i>, con factor ≈3 "
            "(CNN), ≈1.15 (FE, backward solo en clasificador) y ≈2.2 (FT con bloques 4+5). "
            "Es una aproximación documentada, no una medición exacta del autograd.",
            styles["Body2"],
        )
    )

    # Figuras
    story.append(Paragraph("3.3 Gráficas comparativas", styles["H3c"]))
    imgs = []
    p1 = FIG / "val_acc_vs_time.png"
    if p1.exists():
        imgs.append(Image(str(p1), width=5.6 * inch, height=3.4 * inch))
    p2 = FIG / "f1_vs_flops.png"
    if p2.exists():
        imgs.append(Spacer(1, 6))
        imgs.append(Image(str(p2), width=4.6 * inch, height=3.2 * inch))
    if imgs:
        story.append(KeepTogether(imgs))

    # 4. Análisis
    story.append(Paragraph("4. Análisis y conclusiones", styles["H2c"]))
    winner = final.sort_values("test_f1", ascending=False).iloc[0]
    cnn = final[final.family == "CNN"].iloc[0]
    fe = final[final.family == "VGG-FE"].iloc[0]
    ft = final[final.family == "VGG-FT"].iloc[0]
    cnn_drop = float(test[test.family == "CNN"].iloc[0]["f1"]) - float(
        red[red.family == "CNN"].iloc[0]["f1"]
    )
    fe_drop = float(test[test.family == "VGG-FE"].iloc[0]["f1"]) - float(
        red[red.family == "VGG-FE"].iloc[0]["f1"]
    )
    ft_drop = float(test[test.family == "VGG-FT"].iloc[0]["f1"]) - float(
        red[red.family == "VGG-FT"].iloc[0]["f1"]
    )

    story.append(Paragraph("Rendimiento final y trade-off", styles["H3c"]))
    story.append(
        Paragraph(
            f"El mejor F1 de test es <b>{winner['family']}</b> "
            f"({winner['test_f1']:.4f}, accuracy {winner['test_accuracy']:.4f}). "
            f"Respecto a feature extraction gana +{(ft['test_f1']-fe['test_f1']):.4f} F1, "
            f"y frente a la CNN +{(ft['test_f1']-cnn['test_f1']):.4f}. "
            f"Ese margen justifica el costo extra en entrenamiento (≈{ft['total_training_time_s']:.0f} s "
            f"vs ≈{fe['total_training_time_s']:.0f} s en FE y ≈{cnn['total_training_time_s']:.0f} s en CNN) "
            f"y en memoria pico ({ft['peak_gpu_memory_mb']:.0f} MB). "
            f"En inferencia, FE y FT tienen FLOPs casi idénticos (~{fe['flops_per_image']/1e9:.2f} GFLOPs); "
            f"la CNN es dos órdenes de magnitud más ligera ({cnn['flops_per_image']/1e6:.1f} MFLOPs, "
            f"latencia ~{cnn['latency_ms']:.2f} ms).",
            styles["Body2"],
        )
    )

    story.append(Paragraph("Features preentrenadas y redimensionamiento", styles["H3c"]))
    story.append(
        Paragraph(
            "Aunque VGG-16 se preentrenó a 224×224 e ImageNet y CIFAR-10 es 32×32, los filtros "
            "tempranos (bordes, texturas, color) siguen siendo útiles. Redimensionar a 112×112 "
            "no crea detalle nuevo, pero alinea mejor la escala de activaciones; el costo es "
            "≈12.25× más píxeles que la resolución nativa. Usar 224×224 habría multiplicado el "
            "cómputo por ~49× respecto a 32×32, inviables bajo la ventana de este laboratorio.",
            styles["Body2"],
        )
    )

    story.append(Paragraph("Fine-tuning, overfitting y datos reducidos", styles["H3c"]))
    story.append(
        Paragraph(
            f"Descongelar bloques 4+5 (VGG-FT-B) mejoró el F1 de validación frente a solo el "
            f"bloque 5, con learning rates bajos en el backbone (5e-6) para limitar "
            f"<i>catastrophic forgetting</i>. Las curvas muestran convergencia estable sin "
            f"colapso de validación. En el experimento al 10 % de train, la CNN pierde "
            f"{cnn_drop:.3f} F1 absoluto, FE {fe_drop:.3f} y FT {ft_drop:.3f}: con pocos datos "
            f"conviene transfer learning; con datos abundantes una CNN nativa sigue siendo "
            f"atractiva por costo.",
            styles["Body2"],
        )
    )

    story.append(Paragraph("Errores por clase y despliegue", styles["H3c"]))
    story.append(
        Paragraph(
            "Las matrices de confusión concentran errores en pares visualmente ambiguos a baja "
            "resolución: <i>cat↔dog</i>, <i>automobile↔truck</i>, <i>bird↔airplane</i> y "
            "<i>deer↔horse</i>. El fine-tuning reduce estas confusiones pero no las elimina. "
            "Para despliegue en <b>servidor con GPU</b> se recomienda VGG-FT (mejor F1). "
            "En <b>dispositivo móvil</b>, la CNN propia o, preferiblemente, un backbone ligero "
            "como <b>MobileNetV3</b> o <b>EfficientNet-B0</b>, con muchos menos parámetros y "
            "FLOPs que VGG-16.",
            styles["Body2"],
        )
    )

    doc.build(story)
    print("PDF escrito:", ROOT / "Informe_Lab6.pdf")


if __name__ == "__main__":
    main()
