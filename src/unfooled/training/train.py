# flake8: noqa
from __future__ import annotations
import math, os, sys, re, json, time, random, pathlib, logging
import numpy as np
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    # allow import without torch on docs/build
    torch = None
    nn = object
    F = None
    Dataset = object
    DataLoader = object
##############################################################################
# TRAIN / EVAL LOOPS
##############################################################################


def train(num_epochs=EPOCHS):
    for epoch in range(1, num_epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}")
        for x, m_batch, y, paths in pbar:
            x, y, m_batch = x.to(device), y.to(device), m_batch.to(device)

# Evaluation (only change is pmask aggregation)
def evaluate(loader, tta_n=1):
    model.eval()
    all_probs, all_labels, ious_h, ious_s = [], [], [], []
    with torch.no_grad():
        for x, m, y, p in tqdm(loader, desc="eval", leave=False):
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits_list, masks_list = [], []
            for t in range(max(1, int(tta_n))):
                if t == 0:
                    xx = x if x.shape[-1]==IMG_SIZE else F.interpolate(x, (IMG_SIZE, IMG_SIZE))
                    logit, mlogit = model(xx)
                else:
                    pils = denorm_to_pil(x)
                    jitter = []
                    for im in pils:
                        w, h = im.size
                        dw, dh = int(0.02*w), int(0.02*h)
                        left, top = random.randint(0, max(0,dw)), random.randint(0, max(0,dh))
                        im2 = im.crop((left, top, w-left, h-top)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                        im2 = pil_jpeg_roundtrip(im2, quality=random.randint(70, 90))
                        im2 = pil_gamma(im2, gamma=random.uniform(0.95,1.05))
                        jitter.append(im2)
                    xx = norm_batch(tensorize_pil_list(jitter, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
                    logit, mlogit = model(xx)
                if mlogit.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                    mlogit = F.interpolate(mlogit, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
                logits_list.append(logit)
                masks_list.append(torch.sigmoid(mlogit))
