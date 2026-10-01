# Laboratorio 6 — Transfer Learning y Fine-Tuning (CIFAR-10)

Comparación de:
- CNN desde cero (32×32)
- VGG-16 feature extractor (112×112)
- VGG-16 fine-tuning (112×112)

## Entregables
- `Laboratorio6_Transfer_Learning.ipynb` — notebook completo
- `Informe_Lab6.pdf` — informe (≤4 páginas)
- `outputs_lab6/` — métricas, historiales, checkpoints de los mejores modelos
- `figures/` — curvas, matrices de confusión y gráficas comparativas

## Cómo regenerar resultados
```bash
pip install torch torchvision scikit-learn pandas matplotlib fvcore thop tabulate reportlab nbformat
python scripts/run_lab6.py
python scripts/build_notebook.py
python scripts/build_pdf.py
```

## Nota sobre resolución VGG
Se usa **112×112** (no 224×224) por restricciones de cómputo, con la misma resolución en feature extraction y fine-tuning.
