# IncDet‑EWC (Huber EWC + Pseudo‑Annotation)

Use `HuberEWCPlugin` with `PseudoAnnotationPlugin` to mitigate forgetting and leverage confident predictions on past classes.

## Quickstart (SplitMNIST)
```bash
python run_incdet_ewc.py --device auto --train_mb_size 64 --eval_mb_size 64 \
  --train_epochs 1 --lr 0.01 --momentum 0.9 --ewc_lambda 1000.0 --beta 0.5 \
  --confidence_thresh 0.95 --n_experiences 2 --seed 42
```

## CIFAR
```bash
python examples/incdet_ewc_cifar.py --benchmark cifar10 --use_huber 1 --train_mb_size 128 --train_epochs 2
```

## Minimal API
```python
from avalanche.training.plugins import HuberEWCPlugin
from avalanche.training.plugins.pseudo_annotation import PseudoAnnotationPlugin
plugins = [HuberEWCPlugin(ewc_lambda=1000.0, beta=0.5),
           PseudoAnnotationPlugin(confidence_thresh=0.95)]
# pass plugins=plugins into your Naive(...) strategy
```
