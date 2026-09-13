# UnFooled

This codebase was created by Noor Fatima

It exposes importable modules in `src/unfooled/` and runnable CLIs.

## Install (dev)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Use
```bash
# training & evaluation entry points (wire to your loops)
unfooled-train --data ./data --epochs 1
unfooled-eval  --data ./data --checkpoint ckpt.pt
```
