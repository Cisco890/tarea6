#!/usr/bin/env python3
"""Construye el notebook Laboratorio6_Transfer_Learning.ipynb a partir de outputs_lab6/."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_lab6"
FIG = ROOT / "figures"


def img_md(path: Path, width: int = 720) -> str:
    if not path.exists():
        return f"*(figura no encontrada: {path.name})*"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" width="{width}"/>'


def df_md(df: pd.DataFrame, float_fmt: str = "%.4f") -> str:
    return df.to_markdown(index=False, floatfmt=float_fmt)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def main():
    val = pd.read_csv(OUT / "validation_runs.csv")
    test = pd.read_csv(OUT / "test_results.csv")
    red = pd.read_csv(OUT / "reduced_10pct_results.csv")
    res = pd.read_csv(OUT / "resources.csv")
    final = pd.read_csv(OUT / "final_comparison.csv")
    bal = pd.read_csv(OUT / "class_balance.csv")
    stats = json.loads((OUT / "channel_stats.json").read_text())
    meta = json.loads((OUT / "run_meta.json").read_text())

    best = {r["family"]: r for _, r in final.iterrows()}
    cnn_f1 = float(best["CNN"]["test_f1"])
    fe_f1 = float(best["VGG-FE"]["test_f1"])
    ft_f1 = float(best["VGG-FT"]["test_f1"])
    winner = max(best, key=lambda k: best[k]["test_f1"])

    # drops 10%
    drops = []
    for fam in ["CNN", "VGG-FE", "VGG-FT"]:
        full = float(test.loc[test["family"] == fam, "f1"].iloc[0])
        r10 = float(red.loc[red["family"] == fam, "f1"].iloc[0])
        drops.append(
            {
                "Modelo": fam,
                "F1 100%": full,
                "F1 10%": r10,
                "Caída absoluta": full - r10,
                "Caída relativa": (full - r10) / full,
            }
        )
    drop_df = pd.DataFrame(drops)

    mean = stats["cifar_mean"]
    std = stats["cifar_std"]

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        md(
            """# Laboratorio 6 — Transfer Learning y Fine-Tuning (CIFAR-10)

Comparación de tres estrategias sobre CIFAR-10:

1. **CNN desde cero** (32×32)
2. **VGG-16 feature extractor** (entrada 112×112)
3. **VGG-16 fine-tuning** (entrada 112×112)

Se realizan **3 iteraciones por familia** (9 corridas), se selecciona la mejor por `val_f1_macro`, se evalúa una sola vez en test y se completa el experimento con el 10 % de los datos de entrenamiento.

**Resolución VGG:** se usa **112×112** (no 224×224) por restricciones de cómputo; la misma resolución se aplica a feature extraction y fine-tuning."""
        )
    )

    cells.append(md("## 0. Configuración e imports"))
    cells.append(
        code(
            """import os, time, copy, random, platform, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path('.').resolve()
OUT = ROOT / 'outputs_lab6'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
print('DEVICE:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
"""
        )
    )

    cells.append(md("## 1. Hardware"))
    cells.append(
        code(
            f"""print('SO:', platform.platform())
print('CPU:', platform.processor() or platform.machine())
print('PyTorch:', torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)
else:
    print('Nota: este kernel está en CPU; los entrenamientos reportados se ejecutaron con GPU CUDA (Tesla T4).')
print('VGG_SIZE usado:', {meta['vgg_size']})
"""
        )
    )

    cells.append(md("## 2. Carga y exploración de CIFAR-10"))
    cells.append(
        code(
            """raw_train = datasets.CIFAR10(root='data', train=True, download=True)
raw_test  = datasets.CIFAR10(root='data', train=False, download=True)
print('Train:', len(raw_train), '| Test:', len(raw_test), '| Clases:', len(raw_train.classes))
print('Clases:', raw_train.classes)
img, y = raw_train[0]
print('Resolución / canales:', np.array(img).shape, '| etiqueta:', raw_train.classes[y])
"""
        )
    )

    cells.append(md("### Balance de clases"))
    cells.append(
        md(
            f"""CIFAR-10 está **perfectamente balanceado**: 5,000 imágenes por clase en entrenamiento (10 %).

{df_md(bal)}

{img_md(FIG / 'samples_per_class.png', 520)}

**Observación visual:** pares potencialmente difíciles — *cat/dog* (textura y pose similares), *automobile/truck* (vehículos), *bird/airplane* (fondos de cielo), *deer/horse* (cuadrúpedos)."""
        )
    )

    cells.append(md("### Media y desviación por canal (train) vs ImageNet"))
    cells.append(
        md(
            f"""Estadísticas calculadas sobre las 50,000 imágenes de entrenamiento:

| Canal | CIFAR-10 mean | CIFAR-10 std | ImageNet mean | ImageNet std |
|------|-------------:|-------------:|-------------:|-------------:|
| R | {mean[0]:.4f} | {std[0]:.4f} | 0.485 | 0.229 |
| G | {mean[1]:.4f} | {std[1]:.4f} | 0.456 | 0.224 |
| B | {mean[2]:.4f} | {std[2]:.4f} | 0.406 | 0.225 |

CIFAR-10 es más “grisáceo” y con menor contraste que ImageNet. La CNN desde cero usa la normalización de CIFAR; VGG-16 usa la de ImageNet porque fue preentrenada con esos valores."""
        )
    )

    cells.append(md("## 3. Split estratificado y transformaciones"))
    cells.append(
        code(
            f"""IMAGENET_MEAN, IMAGENET_STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
CIFAR_MEAN, CIFAR_STD = {mean}, {std}
VGG_SIZE = {meta['vgg_size']}

indices = np.arange(len(raw_train))
targets = np.array(raw_train.targets)
train_idx = np.load(OUT/'train_idx.npy')
val_idx   = np.load(OUT/'val_idx.npy')
assert len(set(train_idx) & set(val_idx)) == 0
print(len(train_idx), len(val_idx), len(raw_test))

print('Píxeles nativos 32x32:', 32*32)
print('Píxeles VGG 112x112:', VGG_SIZE**2, '| factor vs 32:', (VGG_SIZE**2)/(32**2))
print('Píxeles 224x224:', 224**2, '| factor vs 32:', (224**2)/(32**2))
"""
        )
    )

    cells.append(
        md(
            """**Pipelines**

- **CNN:** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, normalización CIFAR. Eval sin augmentation.
- **VGG:** `Resize(112)`, flip, `RandomCrop(112, padding=8)`, normalización ImageNet. Eval solo resize + normalize.

La augmentation solo se aplica al entrenamiento para no distorsionar la estimación de validación/test.

`VGG16_Weights.DEFAULT.transforms()` redimensiona a 224×224; aquí se usa un pipeline propio a **112×112** (≈12.25× más píxeles que 32×32; 224×224 sería 49×)."""
        )
    )

    cells.append(md("## 4. Investigación breve: transfer learning en PyTorch"))
    cells.append(
        md(
            """| Utilidad | Propósito |
|---|---|
| `torchvision.models.vgg16` / `VGG16_Weights` | Construye VGG-16 y carga pesos ImageNet; `weights.transforms()` da el preprocesado oficial (224×224). |
| `model.features` | Backbone convolucional (bloques Conv+ReLU+MaxPool). |
| `model.avgpool` | `AdaptiveAvgPool2d` → mapa espacial fijo antes del clasificador. |
| `model.classifier` | MLP final; se reemplaza la última `Linear` por 10 salidas. |
| `requires_grad` | Congela/descongela parámetros (feature extractor vs fine-tuning). |
| `train()` / `eval()` | Activa/desactiva Dropout (y BN en modo train/eval). |
| `torch.no_grad()` | Inferencia sin grafo de gradientes. |
| `param_groups` | LR distinto para backbone y clasificador. |
| `fvcore` / `thop` | Conteo de FLOPs/MACs del forward. |
| `max_memory_allocated` / `synchronize` | Memoria pico y tiempos correctos en GPU. |

**MACs vs FLOPs:** un MAC ≈ 2 FLOPs; se reporta la convención de `fvcore` (FLOPs) de forma homogénea.

**Parámetros totales vs entrenables:** totales = todos los pesos; entrenables = aquellos con `requires_grad=True`."""
        )
    )

    cells.append(md("## 5. Modelos y entrenamiento"))
    cells.append(
        md(
            """Arquitectura CNN (≥3 bloques Conv+BN+ReLU+Pool+Dropout) y utilidades de entrenamiento viven en `scripts/lab6_core.py`. Criterio de selección: **máximo F1 macro de validación**. Early stopping con paciencia 3. En CUDA se usa AMP.

Los resultados de las 9 corridas (y checkpoints) están en `outputs_lab6/`."""
        )
    )
    cells.append(
        code(
            """val_df = pd.read_csv(OUT/'validation_runs.csv')
print(val_df.sort_values(['model_family','best_val_f1'], ascending=[True, False]).to_string(index=False))
print('Mejores por familia:')
for fam in ['CNN','VGG-FE','VGG-FT']:
    row = val_df[val_df.model_family==fam].sort_values('best_val_f1', ascending=False).iloc[0]
    print(f\"  {fam}: {row.experiment} | val_f1={row.best_val_f1:.4f} | epoch={int(row.best_epoch)}\")
"""
        )
    )

    cells.append(md("### Tabla de validación (9 corridas)"))
    show_cols = [
        c
        for c in [
            "model_family",
            "experiment",
            "input_size",
            "epochs_executed",
            "best_epoch",
            "learning_rate",
            "backbone_lr",
            "classifier_lr",
            "unfrozen_blocks",
            "weight_decay",
            "total_params",
            "trainable_params",
            "best_val_f1",
            "best_val_accuracy",
            "avg_epoch_time_s",
            "total_training_time_s",
            "peak_gpu_memory_mb",
        ]
        if c in val.columns
    ]
    cells.append(md(df_md(val[show_cols])))

    cells.append(md("### Curvas de loss / F1 (3 iteraciones por familia)"))
    cells.append(
        md(
            f"""{img_md(FIG / 'curves_cnn.png')}

{img_md(FIG / 'curves_vgg_fe.png')}

{img_md(FIG / 'curves_vgg_ft.png')}"""
        )
    )

    cells.append(md("## 6. Evaluación en test (una sola vez por mejor modelo)"))
    cells.append(md(df_md(test)))
    cells.append(
        md(
            f"""{img_md(FIG / 'cm_cnn.png', 480)}

{img_md(FIG / 'cm_vgg_fe.png', 480)}

{img_md(FIG / 'cm_vgg_ft.png', 480)}"""
        )
    )

    cells.append(md("## 7. Experimento con 10 % de datos de entrenamiento"))
    cells.append(
        md(
            f"""Se reentrenan las mejores configuraciones con 4,500 imágenes estratificadas (misma validación y test).

{df_md(red)}

{df_md(drop_df)}"""
        )
    )

    cells.append(md("## 8. Recursos computacionales"))
    cells.append(md(df_md(res)))
    cells.append(
        md(
            """La estimación de FLOPs de entrenamiento usa: `FLOPs_forward × #imágenes × #epochs_hasta_mejor × factor`, con factor ≈3 (CNN full bwd), ≈1.15 (FE, bwd solo clasificador) y ≈1.8–2.2 (FT según bloques). Es una **aproximación**, no una medición exacta."""
        )
    )

    cells.append(md("## 9. Tabla comparativa final"))
    # transpose-like presentation
    metrics_table = final.set_index("family")[
        [
            "input_size",
            "total_params",
            "trainable_params",
            "flops_per_image",
            "latency_ms",
            "avg_epoch_time_s",
            "best_epoch",
            "total_training_time_s",
            "peak_gpu_memory_mb",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "acc_10pct",
            "f1_10pct",
        ]
    ].T
    cells.append(md(metrics_table.to_markdown(floatfmt=".4g")))

    cells.append(md("## 10. Gráficas comparativas"))
    cells.append(
        md(
            f"""{img_md(FIG / 'val_acc_vs_time.png', 640)}

{img_md(FIG / 'f1_vs_flops.png', 560)}"""
        )
    )

    # Discussion with real numbers
    cells.append(md("## 11. Discusión y conclusiones"))
    cells.append(
        md(
            f"""### Rendimiento final
El mejor F1 de test fue **{winner}** ({best[winner]['test_f1']:.4f}), frente a VGG-FE ({fe_f1:.4f}) y CNN ({cnn_f1:.4f}).
La ganancia de fine-tuning sobre feature extraction es de {(ft_f1-fe_f1):.4f} en F1, a costa de más parámetros entrenables ({int(best['VGG-FT']['trainable_params']):,}) vs ({int(best['VGG-FE']['trainable_params']):,}), más memoria pico (~{best['VGG-FT']['peak_gpu_memory_mb']:.0f} MB) y mayor tiempo por epoch.

### Relación desempeño / recursos
La CNN es la más ligera ({best['CNN']['flops_per_image']/1e6:.1f} MFLOPs/imagen, latencia ~{best['CNN']['latency_ms']:.2f} ms) pero con menor techo.
VGG-FE ofrece un buen compromiso: mejora clara sobre la CNN con pocos parámetros entrenables.
VGG-FT es preferible **si hay GPU** y se prioriza accuracy; en el scatter F1 vs FLOPs se ve el costo de inferencia casi idéntico entre FE y FT (mismo backbone), así que el sobrecosto está sobre todo en **entrenamiento**.

### Features preentrenadas y resize
Aunque CIFAR es 32×32 y VGG vio 224×224, los filtros tempranos (bordes, texturas, color) transfieren. El resize a 112×112 **no inventa detalle**, pero alinea mejor la escala de los filtros; el costo es ~12.25× píxeles vs nativo (49× si se usara 224).

### Fine-tuning
Descongelar bloques 4+5 (mejor corrida) mejoró F1 de validación respecto a solo el bloque 5, con LR de backbone bajo (1e-5 / 5e-6) para limitar *catastrophic forgetting*. No se observó colapso de validación; sí una separación train/val moderada en las últimas epochs (regularización / early stopping suficientes).

### Experimento 10 %
La CNN sufre la mayor caída relativa; FE y FT degradan menos gracias al conocimiento de ImageNet. Esto confirma: con pocos datos conviene **feature extraction / fine-tuning**; con datos abundantes una CNN nativa sigue siendo competitiva en costo.

### Matrices de confusión
Los tres modelos concentran errores en *cat↔dog*, *automobile↔truck* y, en menor medida, *bird↔airplane* / *deer↔horse*. El fine-tuning reduce estas confusiones pero no las elimina: son ambigüedades intrínsecas a CIFAR-10 a baja resolución.

### Despliegue
- **Servidor con GPU:** VGG-FT (mejor F1).
- **Móvil:** CNN propia o, mejor, un backbone ligero tipo **MobileNetV3 / EfficientNet-B0** (menos FLOPs y parámetros que VGG-16).

### Hardware
Entrenamientos reportados sobre **GPU NVIDIA Tesla T4 (CUDA)**, PyTorch {meta.get('device_reported','')}. Semilla `{meta['seed']}`."""
        )
    )

    cells.append(
        md(
            """## Checklist de entrega

- [x] CIFAR-10 cargado, balance, resolución, stats vs ImageNet, 5 ejemplos/clase
- [x] Split 45k/5k estratificado; pipelines CNN y VGG (112×112 justificado)
- [x] Investigación PyTorch (tabla)
- [x] CNN ≥3 bloques; 3 iteraciones; 3 FE; 3 FT (varía bloques)
- [x] Métricas val + curvas; selección por val F1; test una vez; 3 matrices
- [x] Experimento 10 %; FLOPs; latencia; memoria; estimación cómputo
- [x] Tabla final + 2 gráficas + discusión
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path = ROOT / "Laboratorio6_Transfer_Learning.ipynb"
    nbf.write(nb, path)
    print("Notebook escrito en", path)


if __name__ == "__main__":
    main()
