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
# UTILITY FUNCTIONS / METRICS
##############################################################################


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# (misc)
#Installs
# %pip -q install --upgrade --no-cache-dir \
  numpy==1.26.4 \
  torch==2.4.0 torchvision==0.19.0 \
  opencv-python-headless==4.9.0.80 \
  "albumentations>=1.4,<1.6" "tqdm>=4.66,<5" \
  "mediapipe>=0.10,<0.11" "huggingface_hub>=0.24,<0.27" \
  "datasets>=3.0,<3.1" "scikit-learn>=1.5,<1.7" "matplotlib>=3.8,<3.10" \
  insightface==0.7.3 onnxruntime==1.18.0

import warnings, numpy as np
warnings.filterwarnings("ignore", message="`rcond` parameter will change", category=FutureWarning, module="insightface")

import numpy as _np, os as _os
print("NumPy:", _np.__version__)
_os.kill(_os.getpid(), 9)

# (misc)
MODE = "FAST"

# (misc)
if MODE == "FAST":
    N_FAKE, N_REAL = 800, 800
    EPOCHS, BATCH_SIZE, TTA_N = 2, 32, 3
else:
    N_FAKE, N_REAL = 4000, 4000
    EPOCHS, BATCH_SIZE, TTA_N = 5, 48, 5

# (misc)
EVAL_IOU_ON_ATTACKS = True

# (misc)
FAKE_DATASET = "OpenRL/DeepFakeFace"
REAL_DATASET = "nielsr/CelebA-faces"
VAL_FRAC, TEST_FRAC = 0.15, 0.15

# (misc)
DATA_ROOT = "/content/unfooled_data_v2_2"
FAKE_DIR = f"{DATA_ROOT}/fake"
REAL_DIR = f"{DATA_ROOT}/real"

# (misc)
print(dict(MODE=MODE, N_FAKE=N_FAKE, N_REAL=N_REAL, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE,
           TTA_N=TTA_N, EVAL_IOU_ON_ATTACKS=EVAL_IOU_ON_ATTACKS))

# (misc)
#Imports & Setup
import os, io, uuid, random, math, json, glob, warnings
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from datasets import load_dataset, disable_caching as hf_disable_caching
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# (misc)
SEED = 2025
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

# (misc)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
Image.MAX_IMAGE_PIXELS = 933120000
hf_disable_caching()

# (misc)
# Silence retinaface FutureWarning spam
warnings.filterwarnings("ignore", message="`rcond` parameter will change",
                        category=FutureWarning, module="insightface")

# (misc)
# ImageNet stats
_IM_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IM_STD  = torch.tensor([0.229, 0.224, 0.225])

# (misc)
def pad_to_square(im: Image.Image, fill=0):
    w, h = im.size
    if w == h: return im
    s = max(w, h)
    out = Image.new("RGB", (s, s), color=(fill, fill, fill))
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out

# (misc)
def pil_jpeg_roundtrip(im: Image.Image, quality=75, subsampling="4:2:0"):
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=int(quality), subsampling=subsampling)
    buf.seek(0); return Image.open(buf).convert("RGB")

# (misc)
def pil_gamma(im: Image.Image, gamma=1.0):
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = np.clip(arr, 1e-6, 1.0) ** float(gamma)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# (misc)
def norm_batch(x: torch.Tensor) -> torch.Tensor:
    mean = _IM_MEAN.to(x.device)[None, :, None, None]
    std  = _IM_STD.to(x.device)[None, :, None, None]
    return (x - mean) / std

# (misc)
def denorm_batch(x: torch.Tensor) -> torch.Tensor:
    mean = _IM_MEAN.to(x.device)[None, :, None, None]
    std  = _IM_STD.to(x.device)[None, :, None, None]
    return x * std + mean

# (misc)
def tensorize_pil_list(pils: list[Image.Image], resize_to: tuple[int,int]|None=None) -> torch.Tensor:
    t = torch.stack([transforms.ToTensor()(im) for im in pils], dim=0)
    if resize_to is not None:
        t = F.interpolate(t, size=resize_to, mode="bilinear", align_corners=False)
    return t

# (misc)
def denorm_to_pil(x: torch.Tensor) -> list[Image.Image]:
    x = denorm_batch(x).clamp(0, 1)
    return [to_pil_image(xx.cpu()) for xx in x]

# (misc)
# @title Cached Hugging Face images locally (no generation)
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

import warnings
warnings.filterwarnings(
    "ignore",
    message="The secret `HF_TOKEN` does not exist in your Colab secrets.",
    category=UserWarning,
    module="huggingface_hub"
)

# (misc)
def cache_hf_images(dataset_id: str, split: str, column: str="image", n: int=1000, out_dir: str="."):
    ds = load_dataset(dataset_id, split=split, streaming=True)
    paths, i = [], 0
    for rec in ds:
        if i >= n: break
        img = rec.get(column, None)
        if img is None:
            img = next(iter(rec.values()))
        try:
            if isinstance(img, Image.Image):
                im = img.convert("RGB")
            elif isinstance(img, dict) and "bytes" in img:
                im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, (bytes, bytearray)):
                im = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                im = Image.fromarray(np.array(img)).convert("RGB")
        except Exception:
            continue
        im = pad_to_square(im).resize((384, 384), Image.BICUBIC)
        fp = os.path.join(out_dir, f"{uuid.uuid4().hex}.jpg")
        im.save(fp, "JPEG", quality=95, subsampling="4:2:0")
        paths.append(fp); i += 1
        if i % 100 == 0: print("  saved", i)
    return paths

# (misc)
fake_paths = cache_hf_images(FAKE_DATASET, "train", "image", N_FAKE, FAKE_DIR)
real_paths = cache_hf_images(REAL_DATASET, "train", "image", N_REAL, REAL_DIR)
print("cached:", len(fake_paths), "fake,", len(real_paths), "real")

import mediapipe as mp
mp_face = mp.solutions.face_detection

# (misc)
def retinaface_bbox(img_rgb):
    try:
        import insightface
        from insightface.app import FaceAnalysis
        if not hasattr(retinaface_bbox, "_app"):
            retinaface_bbox._app = FaceAnalysis(name="buffalo_l")
            retinaface_bbox._app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640,640))
        app = retinaface_bbox._app
        faces = app.get(img_rgb)
        if not faces: return None
        f = max(faces, key=lambda o: (o.bbox[2]-o.bbox[0])*(o.bbox[3]-o.bbox[1]))
        x0,y0,x1,y1 = [int(v) for v in f.bbox]
        h,w = img_rgb.shape[:2]
        x0,y0 = max(0,x0), max(0,y0); x1,y1 = min(w,x1), min(h,y1)
        if x1<=x0 or y1<=y0: return None
        m = np.zeros((h,w), dtype=np.uint8); m[y0:y1, x0:x1] = 1
        return m
    except Exception:
        return None

def robust_face_bbox_mask(im: Image.Image, expand=0.15, min_conf=0.30) -> np.ndarray:
    arr = np.array(im.convert("RGB")); h, w = arr.shape[:2]
    # 1) RetinaFace
    m = retinaface_bbox(arr)
    if m is not None:
        ys, xs = np.where(m>0)
        y0,y1 = ys.min(), ys.max(); x0,x1 = xs.min(), xs.max()
        ew, eh = int((x1-x0)*expand), int((y1-y0)*expand)
        x0,y0 = max(0, x0-ew), max(0, y0-eh); x1,y1 = min(w, x1+ew), min(h, y1+eh)
        out = np.zeros((h,w), dtype=np.uint8); out[y0:y1, x0:x1] = 1
        return out
    # 2) MediaPipe
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_conf) as fd:
        res = fd.process(np.ascontiguousarray(arr))
    if res and res.detections:
        det = max(res.detections, key=lambda d: float(d.score[0]))
        box = det.location_data.relative_bounding_box
        x0 = int((box.xmin - expand) * w); y0 = int((box.ymin - expand) * h)
        x1 = int((box.xmin + box.width + expand) * w); y1 = int((box.ymin + box.height + expand) * h)
        x0,y0 = max(0,x0), max(0,y0); x1,y1 = min(w,x1), min(h,y1)
        if x1>x0 and y1>y0:
            m = np.zeros((h,w), dtype=np.uint8); m[y0:y1, x0:x1] = 1
            return m
    # 3) Center fallback
    s = int(0.7*min(w,h)); cx,cy = w//2, h//2
    x0,y0 = max(0,cx-s//2), max(0,cy-s//2); x1,y1 = min(w,x0+s), min(h,y0+s)
    m = np.zeros((h,w), dtype=np.uint8); m[y0:y1, x0:x1] = 1
    return m

# (misc)
MASK_CACHE_DIR = os.path.join(DATA_ROOT, "mask_cache_v2_2"); os.makedirs(MASK_CACHE_DIR, exist_ok=True)
def _mask_cache_path(img_path: str) -> str:
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(MASK_CACHE_DIR, base + ".npy")

# (misc)
def cached_face_mask(img_path: str, im: Image.Image|None=None) -> np.ndarray:
    cp = _mask_cache_path(img_path)
    if os.path.exists(cp):
        try: return np.load(cp)
        except Exception: pass
    if im is None: im = Image.open(img_path).convert("RGB")
    m = robust_face_bbox_mask(im); np.save(cp, m.astype(np.uint8))
    return m

# (misc)
def soften_mask(m: np.ndarray, sigma=2.0) -> np.ndarray:
    m = m.astype(np.float32); m = cv2.GaussianBlur(m, (0,0), sigmaX=sigma)
    if m.max() > 0: m = m / m.max()
    return m

# (misc)
# @title #Counter-Forensic Attacks (Red-Team Pool)
def attack_jpeg_realign(im: Image.Image):
    w, h = im.size; dx, dy = random.randint(-3,3), random.randint(-3,3)
    canvas = Image.new("RGB", (w+8, h+8), (0,0,0)); canvas.paste(im, (4+dx, 4+dy))
    im2 = canvas.crop((4,4,4+w,4+h))
    return pil_jpeg_roundtrip(im2, quality=random.randint(35,85), subsampling="4:2:0")

# (misc)
def attack_resample_warp(im: Image.Image):
    w, h = im.size; angle = random.uniform(-3,3); scale = random.uniform(0.96,1.04)
    im2 = im.rotate(angle, resample=Image.BICUBIC)
    im2 = im2.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    canv = Image.new("RGB", (w, h), (0,0,0))
    ox = (w - im2.size[0])//2 + random.randint(-2,2); oy = (h - im2.size[1])//2 + random.randint(-2,2)
    canv.paste(im2, (ox, oy)); return canv

# (misc)
def attack_denoise_regrain(im: Image.Image):
    arr = np.array(im.convert("RGB"))
    arr = cv2.fastNlMeansDenoisingColored(arr, None, 7,7,7,21)
    noise = np.random.normal(0, random.uniform(1.0,3.0), arr.shape).astype(np.float32)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2))

# (misc)
def attack_seam_smooth(im: Image.Image):
    m = robust_face_bbox_mask(im)
    if m.mean() == 0: return im
    k = random.choice([3,5,7]); blurred = im.filter(ImageFilter.GaussianBlur(radius=k))
    edges = cv2.Canny((m*255).astype(np.uint8), 0, 1)
    edges = cv2.dilate(edges, np.ones((k,k), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(edges.astype(np.float32), (0,0), sigmaX=k)
    if alpha.max() > 0: alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-6)
    alpha = np.clip(alpha[..., None], 0, 1)
    a = np.array(im).astype(np.float32)/255.0; b = np.array(blurred).astype(np.float32)/255.0
    out = (a*(1-alpha) + b*alpha); out = (out*255.0).clip(0,255).astype(np.uint8)
    return Image.fromarray(out)

# (misc)
def attack_color_gamma(im: Image.Image):
    b = ImageEnhance.Brightness(im).enhance(random.uniform(0.9,1.1))
    c = ImageEnhance.Contrast(b).enhance(random.uniform(0.9,1.1))
    s = ImageEnhance.Color(c).enhance(random.uniform(0.9,1.1))
    return pil_gamma(s, gamma=random.uniform(0.9,1.1))

# (misc)
def attack_social_transcode(im: Image.Image):
    w, h = im.size; target = random.choice([720,640,512,384])
    im2 = im.resize((target, int(h*(target/w))), Image.BICUBIC)
    im2 = ImageOps.pad(im2, (384,384), method=Image.BICUBIC, color=(0,0,0))
    return pil_jpeg_roundtrip(im2, quality=random.randint(55,80))

# (misc)
ATTACK_FUNCS = {
    "jpeg": attack_jpeg_realign,
    "warp": attack_resample_warp,
    "regrain": attack_denoise_regrain,
    "seam": attack_seam_smooth,
    "gamma": attack_color_gamma,
    "transcode": attack_social_transcode,
}
print("attacks:", list(ATTACK_FUNCS.keys()))

# (misc)
# @title #Dataset & loaders
IMG_SIZE = 256

# (misc)
train_aug = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.1), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
])
to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IM_MEAN.tolist(), std=_IM_STD.tolist())
])

# (misc)
def __len__(self): return len(self.samples)

# (misc)
def __getitem__(self, idx):
        p, y = self.samples[idx]
        im = Image.open(p).convert("RGB")
        im_t = train_aug(im) if self.split=="train" else eval_tf(im)
        if y == 1:
            m = cached_face_mask(p, im)
            m = soften_mask(m, sigma=2.0)
        else:
            m = np.zeros((im.height, im.width), dtype=np.float32)
        m = Image.fromarray((m*255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        m = torch.from_numpy(np.array(m).astype(np.float32)/255.0)[None, ...]
        x = to_tensor_norm(im_t); y = torch.tensor(float(y), dtype=torch.float32)
        return x, m, y, p

# (misc)
train_set = DiskDataset(FAKE_DIR, REAL_DIR, "train", SEED, VAL_FRAC, TEST_FRAC)
val_set   = DiskDataset(FAKE_DIR, REAL_DIR, "val",   SEED, VAL_FRAC, TEST_FRAC)
test_set  = DiskDataset(FAKE_DIR, REAL_DIR, "test",  SEED, VAL_FRAC, TEST_FRAC)

# (misc)
print({k: len(v) for k,v in {"train":train_set,"val":val_set,"test":test_set}.items()})

# (misc)
model = UnFooledNet().to(device)
print("params (M):", round(sum(p.numel() for p in model.parameters())/1e6, 2))

# @title # Train
def dice_loss_ps(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2*(p*target).flatten(1).sum(1) + eps
    den = p.flatten(1).sum(1) + target.flatten(1).sum(1) + eps
    return 1.0 - (num/den)

# (misc)
def sobel_edges(x):
    kx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32, device=x.device)
    ky = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32, device=x.device)
    kx = kx.unsqueeze(0); ky = ky.unsqueeze(0)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-6)

# (misc)
def batch_pos_weight(mask_batch, min_w=1.0, max_w=10.0, eps=1e-6):
    a = float(mask_batch.mean().item())
    w = (1.0 - a) / (a + eps)
    return torch.tensor([float(np.clip(w, min_w, max_w))], device=mask_batch.device)

# (misc)
LAMBDA_MASK_BASE = 0.8
EDGE_W, CONS_W, SIZE_W = 0.1, 0.05, 0.15
CLEAN_MASK_W = 0.12

# (misc)
head_params = list(model.res_adapter.parameters()) + list(model.mask_head.parameters())
head_ids = {id(p) for p in head_params}
base_params = [p for p in model.parameters() if id(p) not in head_ids]

# (misc)
optimizer = torch.optim.AdamW(
    [{"params": base_params, "lr": 1e-4, "weight_decay": 1e-4},
     {"params": head_params, "lr": 3e-4, "weight_decay": 1e-4}]
)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

# (misc)
@torch.no_grad()
def red_team_select(x, y, MAXK=2):
    pils = denorm_to_pil(x)
    names = random.sample(list(ATTACK_FUNCS.keys()), k=MAXK)
    scores, tensors, pil_sets = [], [], []
    for n in names:
        ims = [ATTACK_FUNCS[n](im) for im in pils]
        t = tensorize_pil_list(ims, resize_to=(IMG_SIZE, IMG_SIZE)).to(device)
        tx = norm_batch(t); logit, _ = model(tx)
        cls = F.binary_cross_entropy_with_logits(logit, y)
        scores.append(cls.detach()); tensors.append(tx); pil_sets.append(ims)
    worst = int(torch.argmax(torch.stack(scores)))
    return tensors[worst], pil_sets[worst], names[worst]

# clean pass for consistency + aux
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                xx = x if x.shape[-1]==IMG_SIZE else F.interpolate(x,(IMG_SIZE,IMG_SIZE))
                _, mlogit_clean = model(xx)
                pmask_clean = torch.sigmoid(mlogit_clean)

# aux mask loss on clean view (only non-empty)
            m_clean_gt = m_batch.clone()
            if m_clean_gt.shape[-2:] != mlogit_clean.shape[-2:]:
                m_clean_gt = F.interpolate(m_clean_gt, size=mlogit_clean.shape[-2:], mode="nearest")
            valid_clean = (m_clean_gt.sum(dim=(1,2,3)) > 0).float()
            pos_w_c = batch_pos_weight(m_clean_gt)  # positive reweighting
            bce_clean = F.binary_cross_entropy_with_logits(
                mlogit_clean, m_clean_gt, reduction='none', pos_weight=pos_w_c
            ).mean(dim=(1,2,3))
            dice_clean  = dice_loss_ps(mlogit_clean, m_clean_gt)
            mask_clean_loss = (0.5*bce_clean + 0.5*dice_clean)
            mask_clean_loss = (mask_clean_loss * valid_clean).sum() / (valid_clean.sum() + 1e-6)

# (misc)
# worst-of-K (gentler in epoch 1 to help mask)
            with torch.no_grad():
                K = 1 if epoch == 1 else (2 if MODE=="FAST" else 3)  # schedule
                x_att, pils_att, atk_name = red_team_select(x, y, MAXK=K)
            m_att = []
            for im in pils_att:
                mm = robust_face_bbox_mask(im)
                m_att.append(torch.from_numpy(mm.astype(np.float32))[None, ...] / 255.0)
            m_att = torch.stack(m_att, 0).to(device)

# (misc)
lambda_mask = LAMBDA_MASK_BASE * min(1.0, epoch/2)

optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                logit, mlogit = model(x_att)
                pmask = torch.sigmoid(mlogit)

# (misc)
if m_att.shape[-2:] != mlogit.shape[-2:]:
                    m_att = F.interpolate(m_att, size=mlogit.shape[-2:], mode="nearest")
                if pmask_clean.shape[-2:] != pmask.shape[-2:]:
                    pmask_clean = F.interpolate(pmask_clean, size=pmask.shape[-2:], mode="bilinear", align_corners=False)

# (misc)
valid = (m_att.sum(dim=(1,2,3)) > 0).float()

# positive reweighting makes mask less timid
                pos_w = batch_pos_weight(m_att)
                bce_ps  = F.binary_cross_entropy_with_logits(
                    mlogit, m_att, reduction='none', pos_weight=pos_w
                ).mean(dim=(1,2,3))
                dice_ps = dice_loss_ps(mlogit, m_att)
                e_pred  = sobel_edges(pmask).mean(dim=(1,2,3))
                e_tgt   = sobel_edges(m_att).mean(dim=(1,2,3))
                edge_ps = (e_pred - e_tgt).abs()
                area_p  = pmask.mean(dim=(1,2,3))
                area_t  = m_att.mean(dim=(1,2,3))
                size_ps = (area_p - area_t).abs()

# (misc)
mask_loss_ps = 0.6*bce_ps + 0.4*dice_ps + EDGE_W*edge_ps + SIZE_W*size_ps
                mask_loss = (mask_loss_ps * valid).sum() / (valid.sum() + 1e-6)

# (misc)
cons_loss = F.l1_loss(pmask, pmask_clean)
                cls_loss  = F.binary_cross_entropy_with_logits(logit, y)

# (misc)
loss = cls_loss + lambda_mask*mask_loss + CONS_W*cons_loss + CLEAN_MASK_W*mask_clean_loss

# (misc)
scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "atk": atk_name})

# (misc)
val_metrics, _ = evaluate(val_loader, tta_n=1)
        print("Val:", {k: round(v,3) for k,v in val_metrics.items()})

# (misc)
# @title # Evaluation
def mask_iou_bundle(pmask, m_gt, thr=0.3):
    if m_gt.shape[-2:] != pmask.shape[-2:]:
        m_gt = F.interpolate(m_gt, size=pmask.shape[-2:], mode="nearest")
    p = pmask[:,0].clamp(0,1); g = m_gt[:,0].clamp(0,1)
    pbin = (p >= thr); gbin = (g >= 0.5)
    inter = (pbin & gbin).flatten(1).sum(1).float()
    union = (pbin | gbin).flatten(1).sum(1).float() + 1e-6
    iou_hard = (inter / union)
    inter_s = (p * g).flatten(1).sum(1)
    union_s = (p + g - p*g).flatten(1).sum(1) + 1e-6
    iou_soft = (inter_s / union_s)
    return iou_hard, iou_soft

def ece_binary(labels, probs, n_bins=15):
    labels = np.asarray(labels).astype(int)
    probs  = np.asarray(probs).astype(float)
    yhat   = (probs >= 0.5).astype(int)
    conf   = np.where(yhat==1, probs, 1.0 - probs)  # confidence of predicted class
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < (bins[i+1] if i < n_bins-1 else conf <= bins[i+1]))
        if m.sum()==0: continue
        acc = (yhat[m]==labels[m]).mean()
        ece += m.mean() * abs(acc - conf[m].mean())
    return float(ece)

# (misc)
# mean for classification, MAX for mask (prevents dilution)
            logit_agg = torch.stack(logits_list, 0).mean(0)
            pmask_agg = torch.stack(masks_list, 0).amax(0)

probs = torch.sigmoid(logit_agg).detach().cpu().numpy()
            all_probs += probs.tolist(); all_labels += y.cpu().numpy().tolist()

# (misc)
i_h, i_s = mask_iou_bundle(pmask_agg, m, thr=0.3)
            for i in range(y.shape[0]):
                if (y[i] >= 0.5) and (m[i].sum() > 0):
                    ious_h.append(float(i_h[i].item()))
                    ious_s.append(float(i_s[i].item()))

auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    ap  = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    acc = float(np.mean(((np.array(all_probs) >= 0.5).astype(int) == np.array(all_labels)).astype(np.float32)))
    ece = ece_binary(all_labels, all_probs, n_bins=15)
    iou_h_mean = float(np.mean(ious_h)) if len(ious_h) > 0 else float("nan")
    iou_s_mean = float(np.mean(ious_s)) if len(ious_s) > 0 else float("nan")
    return {"AUC": auc, "AP": ap, "ACC": acc, "ECE": ece}, (all_labels, all_probs)
    #return {"AUC": auc, "AP": ap, "ACC": acc, "ECE": ece, "IoU@0.3": iou_h_mean, "IoU_soft": iou_s_mean}, (all_labels, all_probs)

def expected_calibration_error(labels, probs, n_bins=10):
    labels = np.array(labels).astype(int); probs = np.array(probs)
    bins = np.linspace(0,1,n_bins+1); ece=0.0; total=len(labels)
    for i in range(n_bins):
        m = (probs>=bins[i]) & (probs<bins[i+1] if i<n_bins-1 else probs<=bins[i+1])
        if m.sum()==0: continue
        conf = probs[m].mean(); acc = (labels[m]==(probs[m]>=0.5)).mean()
        ece += (m.sum()/total)*abs(acc-conf)
    return float(ece)

def eval_attack(loader, attack_name: str, with_iou=True):
    model.eval()
    all_probs, all_labels = [], []
    ious_h, ious_s = [], []
    with torch.no_grad():
        for x, m, y, p in tqdm(loader, desc=f"atk:{attack_name}", leave=False):
            x, y = x.to(device), y.to(device)
            pils = denorm_to_pil(x)
            ims = [ATTACK_FUNCS[attack_name](im) for im in pils]
            xx = norm_batch(tensorize_pil_list(ims, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
            logit, mlogit = model(xx)
            probs = torch.sigmoid(logit).detach().cpu().numpy()
            all_probs += probs.tolist(); all_labels += y.cpu().numpy().tolist()

if with_iou:
                m2 = []
                for im in ims:
                    mm = robust_face_bbox_mask(im)
                    m2.append(torch.from_numpy(mm.astype(np.float32))[None,...]/255.0)
                m2 = torch.stack(m2,0).to(device)
                pmask_t = torch.sigmoid(mlogit)
                ih, is_ = mask_iou_bundle(pmask_t, m2, thr=0.3)
                for i in range(y.shape[0]):
                    if (y[i] >= 0.5) and (m2[i].sum() > 0):
                        ious_h.append(float(ih[i].item())); ious_s.append(float(is_[i].item()))

auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels))>1 else float('nan')
    acc = float(np.mean(((np.array(all_probs)>=0.5).astype(int) == np.array(all_labels)).astype(np.float32)))
    ece = ece_binary(all_labels, all_probs, n_bins=15)
    out = {"AUC":auc, "ACC":acc, "ECE":ece}
    if with_iou:
        out["IoU@0.3"] = float(np.mean(ious_h)) if len(ious_h)>0 else float('nan')
        out["IoU_soft"] = float(np.mean(ious_s)) if len(ious_s)>0 else float('nan')
    return out

def eval_surveillance(loader):
    def low_light(im):
        im = ImageEnhance.Brightness(im).enhance(0.6)
        im = ImageEnhance.Contrast(im).enhance(0.9)
        im = im.filter(ImageFilter.GaussianBlur(radius=1.0))
        im = pil_jpeg_roundtrip(im, quality=40)
        return im
    model.eval(); all_probs=[]; all_labels=[]
    with torch.no_grad():
        for x, m, y, p in tqdm(loader, desc="surveillance", leave=False):
            pils = denorm_to_pil(x.to(device))
            ims = [low_light(im) for im in pils]
            xx = norm_batch(tensorize_pil_list(ims, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
            logit, _ = model(xx)
            probs = torch.sigmoid(logit).detach().cpu().numpy()
            all_probs += probs.tolist(); all_labels += y.cpu().numpy().tolist()
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels))>1 else float('nan')
    acc = float(np.mean(((np.array(all_probs)>=0.5).astype(int) == np.array(all_labels)).astype(np.float32)))
    ece = ece_binary(all_labels, all_probs, n_bins=15)
    return {"AUC":auc, "ACC":acc, "ECE":ece}

# (misc)
# @title #Run: train then evaluate
train(EPOCHS)

# (misc)
clean_metrics, (labels, probs) = evaluate(test_loader, tta_n=TTA_N)
print("Test (clean):", {k: round(v,3) for k,v in clean_metrics.items()})

# (misc)
results = {"clean": clean_metrics}
for a in ATTACK_FUNCS.keys():
    results[a] = eval_attack(test_loader, a)
    print(a, {k: round(v,3) for k,v in results[a].items()})

worst_acc = min(v["ACC"] for k,v in results.items())
worst_auc = min(v["AUC"] for k,v in results.items())
delta_auc = {k: v["AUC"] - results["clean"]["AUC"] for k,v in results.items() if k!="clean"}
print("\nSummary:", {"worst_ACC": round(worst_acc,3), "worst_AUC": round(worst_auc,3),
                     "ΔAUC": {k: round(d,3) for k,d in delta_auc.items()}})

# (misc)
print("Surveillance:", {k: round(v,3) for k,v in eval_surveillance(test_loader).items()})

# @title #Metrics roc_curve, f1_score, matthews_corrcoef, log_loss
from sklearn.metrics import roc_curve, f1_score, matthews_corrcoef, log_loss

# (misc)
def compute_extra_metrics(labels, probs, tau=0.5):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    if len(np.unique(y)) < 2:
        return {"note": "labels are single-class; metrics undefined"}

# EER and threshold@EER
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1.0 - tpr
    # find point closest to FPR == FNR
    i = np.argmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[i] + fnr[i])
    tau_eer = thr[i]

# (misc)
# APCER/BPCER/HTER at chosen threshold (tau or tau_eer)
    def apcer_bpcer_at(t):
        yhat = (p >= t).astype(int)
        apcer = np.mean((yhat == 0) & (y == 1))  # attack wrongly accepted as bona fide
        bpcer = np.mean((yhat == 1) & (y == 0))  # bona fide wrongly rejected
        hter = 0.5*(apcer + bpcer)
        return apcer, bpcer, hter

# (misc)
apcer05, bpcer05, hter05   = apcer_bpcer_at(tau)
    apcerE,  bpcerE,  hterE    = apcer_bpcer_at(tau_eer)

# (misc)
# Low-FPR operating points
    def tpr_at_fpr(target):
        # Pick the max TPR with FPR <= target
        ok = np.where(fpr <= target)[0]
        return float(tpr[ok].max()) if len(ok) > 0 else float("nan")

# (misc)
tpr_fpr1e2 = tpr_at_fpr(1e-2)
    tpr_fpr1e3 = tpr_at_fpr(1e-3)

# (misc)
# Thresholded classification metrics (at tau)
    yhat_tau = (p >= tau).astype(int)
    f1 = f1_score(y, yhat_tau)
    mcc = matthews_corrcoef(y, yhat_tau)

brier = float(np.mean((p - y)**2))
    nll = float(log_loss(y, p, labels=[0,1]))

return {
        "EER": float(eer), "tau@EER": float(tau_eer),
        "APCER@0.5": float(apcer05), "BPCER@0.5": float(bpcer05), "HTER@0.5": float(hter05),
        "APCER@EER": float(apcerE),  "BPCER@EER": float(bpcerE),  "HTER@EER": float(hterE),
        "TPR@FPR=1e-2": tpr_fpr1e2, "TPR@FPR=1e-3": tpr_fpr1e3,
        "F1@0.5": float(f1), "MCC@0.5": float(mcc),
        "Brier": brier, "NLL": nll,
    }

# (misc)
clean_metrics, (labels, probs) = evaluate(test_loader, tta_n=TTA_N)

# (misc)
print(compute_extra_metrics(labels, probs, tau=0.5))

# (misc)
# @title DET curve (FNR vs FPR)

def plot_det(labels, probs, title="DET curve"):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    if len(np.unique(y)) < 2:
        print("DET undefined (single-class labels)."); return
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1.0 - tpr
    plt.figure()
    plt.semilogx(fpr, fnr, lw=2)
    plt.xlabel("FPR"); plt.ylabel("FNR"); plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(1e-4, 1.0); plt.ylim(0.0, 1.0)
    plt.show()

# (misc)
plot_det(labels, probs, title="DET — Clean")

# (misc)
# @title Risk–Coverage (abstention) + AURC
def risk_coverage(labels, probs):
    """
    Risk = error rate among kept samples, Coverage = kept fraction.
    Sort by confidence and sweep a rejection threshold.
    Returns dict with AURC and arrays for plotting.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat==1, p, 1.0 - p)  # predicted-class confidence

# (misc)
order = np.argsort(-conf)             # high conf first
    y_ord, yhat_ord, conf_ord = y[order], yhat[order], conf[order]

# (misc)
cov, risk = [], []
    correct = 0
    for k in range(1, len(y_ord)+1):
        correct += int(yhat_ord[k-1] == y_ord[k-1])
        cov.append(k / len(y_ord))
        risk.append(1.0 - correct / k)

# (misc)
cov = np.array(cov); risk = np.array(risk)
    aurc = float(np.trapz(risk, cov))  # lower is better
    return {"AURC": aurc, "coverage": cov, "risk": risk}

# (misc)
def plot_risk_coverage(rc, title="Risk–Coverage"):
    plt.figure()
    plt.plot(rc["coverage"], rc["risk"], lw=2)
    plt.xlabel("Coverage"); plt.ylabel("Risk (error rate)")
    plt.title(f"{title} • AURC={rc['AURC']:.4f}")
    plt.grid(True, alpha=0.3); plt.ylim(0,1); plt.xlim(0,1)
    plt.show()

# (misc)
rc = risk_coverage(labels, probs); plot_risk_coverage(rc, title="Clean")

# (misc)
# @title #Qualitative: Predictions & Heatmaps
def overlay_heatmap(img_pil, mask01, cmap_name="turbo", alpha=0.45, alpha_pow=0.6, thr=0.02):
    m = np.asarray(mask01, dtype=np.float32)
    if m.ndim == 3: m = m.squeeze()
    m = np.clip(m, 0.0, 1.0)
    w, h = img_pil.size
    m_img = Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    m_img = np.asarray(m_img, dtype=np.float32) / 255.0
    cmap = mpl.colormaps.get_cmap(cmap_name)
    heat = cmap(m_img)[..., :3]
    a = np.where(m_img >= thr, (m_img ** float(alpha_pow)) * float(alpha), 0.0)[..., None]
    base = np.asarray(img_pil).astype(np.float32) / 255.0
    out = base * (1.0 - a) + heat * a
    return np.clip(out, 0.0, 1.0)

# (misc)
@torch.no_grad()
def visualize(loader, n=6, tta_n=1, show_overlay=True, cmap_pred="turbo"):
    model.eval()
    xs, ms, ys, ps = next(iter(loader))
    n = min(n, xs.size(0))
    xs = xs[:n].to(device); ms = ms[:n].to(device); ys = ys[:n]

# forward (TTA for masks: MAX aggregation to avoid dilution)
    logits_list, masks_list = [], []
    for t in range(max(1,int(tta_n))):
        if t == 0:
            xx = xs if xs.shape[-1]==IMG_SIZE else F.interpolate(xs,(IMG_SIZE,IMG_SIZE))
            logit, mlogit = model(xx)
        else:
            pils = denorm_to_pil(xs)
            jitter = []
            for im in pils:
                w,h = im.size
                dw, dh = int(0.02*w), int(0.02*h)
                left, top = random.randint(0, max(0,dw)), random.randint(0, max(0,dh))
                im2 = im.crop((left, top, w-left, h-top)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                jitter.append(im2)
            xx = norm_batch(tensorize_pil_list(jitter, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
            logit, mlogit = model(xx)
        if mlogit.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            mlogit = F.interpolate(mlogit, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        logits_list.append(logit)
        masks_list.append(torch.sigmoid(mlogit))

probs = torch.sigmoid(torch.stack(logits_list,0).mean(0)).cpu().numpy()
    pmask = torch.stack(masks_list,0).amax(0)              # [B,1,H,W]  (amax avoids washing out)
    ims = denorm_to_pil(xs)

# (misc)
# resize GT & pred mask to image size for plotting
    w0, h0 = ims[0].size
    pmask_vis = F.interpolate(pmask, size=(h0, w0), mode="bilinear", align_corners=False)  # [B,1,h,w]
    ms_vis    = F.interpolate(ms,    size=(h0, w0), mode="nearest")                        # [B,1,h,w]

# (misc)
cols = n
    rows = 3  # image | weak-GT | pred-heat
    plt.figure(figsize=(4*cols, 4*rows))
    k = 1
    for i in range(n):
        # 1) image
        ax = plt.subplot(rows, cols, k); k += 1
        ax.imshow(ims[i]); ax.axis("off")
        ax.set_title(f"label={int(ys[i].item())} • pred={'FAKE' if probs[i]>=0.5 else 'REAL'} ({probs[i]:.2f})")

# (misc)
# 2) weak GT (grayscale for clarity)
        ax = plt.subplot(rows, cols, k); k += 1
        ax.imshow(ms_vis[i,0].cpu(), vmin=0, vmax=1, cmap="gray"); ax.axis("off"); ax.set_title("weak GT")

# (misc)
# 3) predicted mask (either raw heatmap or overlay)
        ax = plt.subplot(rows, cols, k); k += 1
        if show_overlay:
            vis = overlay_heatmap(ims[i], pmask_vis[i,0].cpu().numpy(), cmap_name=cmap_pred, alpha=0.45, alpha_pow=0.6, thr=0.02)
            ax.imshow(vis)
            ax.set_title("overlay (pred)")
        else:
            ax.imshow(pmask_vis[i,0].cpu(), vmin=0, vmax=1, cmap=cmap_pred)
            ax.set_title("pred prob")
        ax.axis("off")

# (misc)
sm = mpl.cm.ScalarMappable(cmap=mpl.colormaps.get_cmap(cmap_pred), norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.tight_layout(); plt.show()

# (misc)
visualize(test_loader, n=min(12, BATCH_SIZE), tta_n=3, show_overlay=True, cmap_pred="turbo")

# (misc)
visualize(test_loader, n=min(9, BATCH_SIZE), tta_n=3, show_overlay=True, cmap_pred="turbo")

# (misc)
visualize(test_loader, n=min(6, BATCH_SIZE), tta_n=3, show_overlay=True, cmap_pred="turbo")

# (misc)
def overlay_heatmap(img_pil, mask01, cmap_name="turbo", alpha=0.45, alpha_pow=0.6, thr=0.02):
    """
    img_pil : PIL.Image (RGB)
    mask01  : HxW float in [0,1] (will be resized to the image size if needed)
    alpha   : max opacity of heatmap
    alpha_pow : raise mask to this power before mixing (boosts mid/high probs)
    thr     : values below this are transparent
    """
    m = np.asarray(mask01, dtype=np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    m = np.clip(m, 0.0, 1.0)

# (misc)
w, h = img_pil.size
    if m.shape[:2] != (h, w):
        m_img = Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        m = np.asarray(m_img, dtype=np.float32) / 255.0

# (misc)
cmap = mpl.colormaps.get_cmap(cmap_name)
    heat = cmap(m)[..., :3]  # drop alpha

# (misc)
# Per-pixel alpha from mask (threshold + power curve)
    a = np.where(m >= thr, (m ** float(alpha_pow)) * float(alpha), 0.0)[..., None]

# (misc)
# Blend
    base = np.asarray(img_pil).astype(np.float32) / 255.0
    out = base * (1.0 - a) + heat * a
    return np.clip(out, 0.0, 1.0)

# (misc)
@torch.no_grad()
def visualize_overlays(loader, n=6, cmap_pred="turbo", cmap_gt="Greens", tta_n=1):
    """
    Shows: original | weak-GT overlay | prediction overlay
    - Uses mean over TTA for logits (classification) and MAX over TTA for masks (keeps peaks).
    """
    model.eval()
    xs, ms, ys, ps = next(iter(loader))
    n = min(n, xs.size(0))
    xs = xs[:n].to(device); ms = ms[:n].to(device); ys = ys[:n]

# (misc)
# Forward (+ optional light TTA)
    logits_list, masks_list = [], []
    for t in range(max(1, int(tta_n))):
        if t == 0:
            xx = xs if xs.shape[-1] == IMG_SIZE else F.interpolate(xs, (IMG_SIZE, IMG_SIZE))
            logit, mlogit = model(xx)
        else:
            # cheap jitter TTA
            pils = denorm_to_pil(xs)
            jitter = []
            for im in pils:
                w, h = im.size
                dw, dh = int(0.02*w), int(0.02*h)
                left  = np.random.randint(0, max(1, dw)+1)
                top   = np.random.randint(0, max(1, dh)+1)
                im2 = im.crop((left, top, w-left, h-top)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                jitter.append(im2)
            xx = norm_batch(tensorize_pil_list(jitter, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
            logit, mlogit = model(xx)

# (misc)
if mlogit.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            mlogit = F.interpolate(mlogit, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

logits_list.append(logit)
        masks_list.append(torch.sigmoid(mlogit))  # (B,1,H,W)

probs = torch.sigmoid(torch.stack(logits_list, 0).mean(0)).cpu().numpy()
    pmask = torch.stack(masks_list, 0).amax(0)  # MAX for masks to avoid dilution

# (misc)
ims = denorm_to_pil(xs)
    w0, h0 = ims[0].size

# (misc)
# Resize GT & pred mask to image size for plotting
    pmask_vis = F.interpolate(pmask, size=(h0, w0), mode="bilinear", align_corners=False)  # (B,1,h,w)
    ms_vis    = F.interpolate(ms,    size=(h0, w0), mode="nearest")                        # (B,1,h,w)

# (misc)
cols = n
    rows = 3
    fig = plt.figure(figsize=(4*cols, 4*rows))
    k = 1
    for i in range(n):
        # 1) original image
        ax = plt.subplot(rows, cols, k); k += 1
        ax.imshow(ims[i]); ax.axis("off")
        ax.set_title(f"label={int(ys[i].item())} • pred={'FAKE' if probs[i]>=0.5 else 'REAL'} ({probs[i]:.2f})")

# (misc)
# 2) weak GT overlay
        gt_overlay = overlay_heatmap(ims[i], ms_vis[i,0].cpu().numpy(),
                                     cmap_name=cmap_gt, alpha=0.6, alpha_pow=1.0, thr=0.02)
        ax = plt.subplot(rows, cols, k); k += 1
        ax.imshow(gt_overlay); ax.axis("off"); ax.set_title("weak GT (overlay)")

# (misc)
# 3) prediction overlay
        pred_overlay = overlay_heatmap(ims[i], pmask_vis[i,0].cpu().numpy(),
                                       cmap_name=cmap_pred, alpha=0.45, alpha_pow=0.6, thr=0.02)
        ax = plt.subplot(rows, cols, k); k += 1
        ax.imshow(pred_overlay); ax.axis("off"); ax.set_title("pred heatmap (overlay)")

# (misc)
sm = mpl.cm.ScalarMappable(cmap=mpl.colormaps.get_cmap(cmap_pred),
                               norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.tight_layout(); plt.show()

# (misc)
visualize_overlays(test_loader, n=min(12, BATCH_SIZE), cmap_pred="turbo", cmap_gt="Greens", tta_n=3)

# (misc)
visualize_overlays(test_loader, n=min(9, BATCH_SIZE), cmap_pred="turbo", cmap_gt="Greens", tta_n=3)

# (misc)
visualize_overlays(test_loader,n=min(6, BATCH_SIZE), cmap_pred="turbo", cmap_gt="Greens", tta_n=3)

# (misc)
# @title Visuals: Overlays + all-attacks Grid
def overlay_heatmap(img_pil, mask01, cmap_name="turbo", alpha=0.45, alpha_pow=0.6, thr=0.02):
    m = np.asarray(mask01, dtype=np.float32)
    if m.ndim == 3: m = m.squeeze()
    m = np.clip(m, 0.0, 1.0)
    w, h = img_pil.size
    m_img = Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    m_img = np.asarray(m_img, dtype=np.float32) / 255.0
    cmap = mpl.colormaps.get_cmap(cmap_name)
    heat = cmap(m_img)[..., :3]
    a = np.where(m_img >= thr, (m_img ** float(alpha_pow)) * float(alpha), 0.0)[..., None]
    base = np.asarray(img_pil).astype(np.float32) / 255.0
    out = base * (1.0 - a) + heat * a
    return np.clip(out, 0.0, 1.0)

# (misc)
@torch.no_grad()
def heatmaps_all_attacks(loader, n=4, attacks=None, include_clean=True, cmap_name="turbo", seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    x, m, y, paths = next(iter(loader))
    sel = np.arange(len(y))
    if len(sel) > n: sel = rng.choice(sel, size=n, replace=False)
    xs = x[sel].to(device)
    pils_clean = denorm_to_pil(xs)

# (misc)
ordered = ["jpeg", "warp", "regrain", "seam", "gamma", "transcode"]
    atk_list = [a for a in (attacks or ordered) if a in ATTACK_FUNCS and a in ordered]
    preds = {}

if include_clean:
        xx = xs if xs.shape[-1]==IMG_SIZE else F.interpolate(xs, (IMG_SIZE, IMG_SIZE))
        logit, mlogit = model(xx)
        preds["clean"] = {
            "probs": torch.sigmoid(logit).cpu().numpy(),
            "pmask": torch.sigmoid(mlogit).cpu().numpy()[:, 0],
            "pils":  pils_clean,
        }
    for a in atk_list:
        pils_a = [ATTACK_FUNCS[a](im) for im in pils_clean]
        t = tensorize_pil_list(pils_a, resize_to=(IMG_SIZE, IMG_SIZE)).to(device)
        xx = norm_batch(t)
        logit, mlogit = model(xx)
        preds[a] = {
            "probs": torch.sigmoid(logit).cpu().numpy(),
            "pmask": torch.sigmoid(mlogit).cpu().numpy()[:, 0],
            "pils":  pils_a,
        }

# (misc)
conds = (["clean"] if include_clean else []) + atk_list
    rows, cols = len(sel), len(conds)
    fig = plt.figure(figsize=(4*cols, 4*rows))
    k = 1
    for i in range(rows):
        for c in conds:
            img = preds[c]["pils"][i]
            pm  = preds[c]["pmask"][i]
            prob = preds[c]["probs"][i]
            vis = overlay_heatmap(img, pm, cmap_name=cmap_name, alpha=0.45, alpha_pow=0.6, thr=0.02)
            ax = plt.subplot(rows, cols, k); k += 1
            ax.imshow(vis); ax.axis("off")
            ax.set_title(f"{c} → {'FAKE' if prob>=0.5 else 'REAL'} ({prob:.2f})", fontsize=11)

# (misc)
sm = mpl.cm.ScalarMappable(cmap=mpl.colormaps.get_cmap(cmap_name), norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.tight_layout(); plt.show()

# (misc)
heatmaps_all_attacks(test_loader, n=12)

# (misc)
heatmaps_all_attacks(test_loader, n=24)

from itertools import islice
import math

# (misc)
def visualize_batchidx(loader, batch_idx=0, n=6, **viz_kwargs):
    # wrap around if idx is larger than num batches
    num_batches = math.ceil(len(loader.dataset) / loader.batch_size)
    batch_idx = batch_idx % num_batches
    xs, ms, ys, ps = next(islice(iter(loader), batch_idx, None))
    tmp_loader = [(xs, ms, ys, ps)]
    visualize_overlays(tmp_loader, n=min(n, xs.size(0)), **viz_kwargs)

# (misc)
visualize_batchidx(test_loader, batch_idx=5, n=6, tta_n=3)

# (misc)
visualize_batchidx(test_loader, batch_idx=5, n=9, tta_n=3)

# (misc)
# @title # REAL vs FAKE visual check

# (misc)
def _overlay_heatmap_red(pil_img: Image.Image, mask01: np.ndarray, alpha=0.35):
    """Blend a red heatmap (mask01 in [0,1]) over the PIL image."""
    m = np.asarray(mask01, dtype=np.float32)
    if m.ndim == 3: m = m.squeeze()
    m = np.clip(m, 0.0, 1.0)
    # resize to image size
    w, h = pil_img.size
    if m.shape[:2] != (h, w):
        m = np.array(Image.fromarray((m*255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    # build red overlay
    heat = np.stack([m, np.zeros_like(m), np.zeros_like(m)], axis=-1)  # R,G,B
    base = np.asarray(pil_img).astype(np.float32) / 255.0
    a = (m >= 0.02).astype(np.float32) * alpha
    a = a[..., None]
    out = base * (1.0 - a) + heat * a
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

# (misc)
def _hard_iou(pmask01: np.ndarray, gt01: np.ndarray, thr=0.3):
    """Hard IoU at threshold thr; returns np.nan if GT empty."""
    p = (pmask01 >= thr).astype(np.uint8)
    g = (gt01 >= thr).astype(np.uint8)
    if g.sum() == 0: return np.nan
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return float(inter / (union + 1e-6))

# (misc)
@torch.no_grad()
def visual_real_vs_fake(loader, n_per_class=4, tta_n=1, title="REAL vs FAKE — prediction, confidence, and heatmap"):
    model.eval()

# (misc)
# up to n reals + n fakes
    buf_real, buf_fake = [], []
    for x, m, y, paths in loader:
        for i in range(len(y)):
            if y[i].item() < 0.5 and len(buf_real) < n_per_class:
                buf_real.append((x[i], m[i], y[i], paths[i]))
            elif y[i].item() >= 0.5 and len(buf_fake) < n_per_class:
                buf_fake.append((x[i], m[i], y[i], paths[i]))
        if len(buf_real) == n_per_class and len(buf_fake) == n_per_class:
            break

# (misc)
samples = buf_real + buf_fake
    if len(samples) == 0:
        print("No samples found in loader."); return

# (misc)
xs = torch.stack([s[0] for s in samples]).to(device)
    ms = torch.stack([s[1] for s in samples]).to(device)  # weak GT mask
    ys = torch.stack([s[2] for s in samples])             # 0 real, 1 fake
    paths = [s[3] for s in samples]
    B = xs.size(0)

# (misc)
# forward (+ optional TTA): mean logits, MAX masks
    logits_list, masks_list = [], []
    for t in range(max(1, int(tta_n))):
        if t == 0:
            xx = xs if xs.shape[-1] == IMG_SIZE else F.interpolate(xs, (IMG_SIZE, IMG_SIZE))
            logit, mlogit = model(xx)
        else:
            # light jitter TTA
            pils = denorm_to_pil(xs)
            jitter = []
            for im in pils:
                w, h = im.size
                dw, dh = int(0.02*w), int(0.02*h)
                left  = np.random.randint(0, max(1, dw)+1)
                top   = np.random.randint(0, max(1, dh)+1)
                im2 = im.crop((left, top, w-left, h-top)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                jitter.append(im2)
            xx = norm_batch(tensorize_pil_list(jitter, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
            logit, mlogit = model(xx)

# (misc)
if mlogit.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            mlogit = F.interpolate(mlogit, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

logits_list.append(logit)
        masks_list.append(torch.sigmoid(mlogit))  # (B,1,H,W)

probs = torch.sigmoid(torch.stack(logits_list, 0).mean(0)).cpu().numpy()   # (B,)
    pmask = torch.stack(masks_list, 0).amax(0)[:, 0].cpu().numpy()            # (B,H,W)  MAX keeps peaks
    pils = denorm_to_pil(xs)

# (misc)
cols = B
    rows = 1
    fig = plt.figure(figsize=(4*cols, 5))
    for i in range(B):
        ax = plt.subplot(rows, cols, i+1)
        y_true = int(ys[i].item() >= 0.5)
        prob   = float(probs[i])
        pred   = int(prob >= 0.5)

# (misc)
# overlay red heatmap
        vis = _overlay_heatmap_red(pils[i], pmask[i], alpha=0.35)
        ax.imshow(vis); ax.axis("off")

# (misc)
title = f"{'FAKE' if pred else 'REAL'} ({prob:.2f}) • GT={'FAKE' if y_true==1 else 'REAL'}"
        ax.set_title(title, fontsize=11)

# (misc)
# green frame if correct, red otherwise
        color = "#23c552" if pred == y_true else "#e63946"
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(3)

# (misc)
plt.suptitle(title if cols == 1 else "REAL vs FAKE — prediction, confidence, and heatmap",
                 fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()

# (misc)
visual_real_vs_fake(test_loader, n_per_class=4, tta_n=3)

# (misc)
visual_real_vs_fake(test_loader, n_per_class=2, tta_n=3)

# (misc)
#@title # UnFooled Figures

# (misc)
ATTACK_COLORS = {
    "jpeg":       "#377eb8",
    "warp":       "#ff7f00",
    "regrain":    "#4daf4a",
    "seam":       "#e41a1c",
    "gamma":      "#984ea3",
    "transcode":  "#a65628",
}
def _color_for(name, i):
    return ATTACK_COLORS.get(name, plt.cm.tab10(i % 10))

# eval-with-labels (clean or attack)
@torch.no_grad()
def _eval_with_labels(loader, attack_name=None, tta_n=1, include_masks=False):
    """
    Returns dict: { 'labels':[], 'probs':[], 'auc':float, 'acc':float }
    If include_masks=True also returns 'pmasks' (B,H,W) per batch (used rarely).
    """
    model.eval()
    all_labels, all_probs = [], []
    for x, m, y, _ in loader:
        x, y = x.to(device), y.to(device)

# (misc)
logits_list = []
        for t in range(max(1,int(tta_n))):
            if t == 0:
                if attack_name is None:
                    xx = x if x.shape[-1]==IMG_SIZE else F.interpolate(x,(IMG_SIZE,IMG_SIZE))
                else:
                    pils = denorm_to_pil(x)
                    ims = [ATTACK_FUNCS[attack_name](im) for im in pils]
                    xx = norm_batch(tensorize_pil_list(ims, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
                logit, _ = model(xx)
            else:
                pils = denorm_to_pil(x)
                jitter = []
                for im in pils:
                    w,h = im.size
                    dw,dh = int(0.02*w), int(0.02*h)
                    left  = np.random.randint(0, max(1,dw)+1)
                    top   = np.random.randint(0, max(1,dh)+1)
                    im2 = im.crop((left,top,w-left,h-top)).resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC)
                    jitter.append(im2)
                xx = norm_batch(tensorize_pil_list(jitter, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
                logit, _ = model(xx)
            logits_list.append(logit)

probs = torch.sigmoid(torch.stack(logits_list,0).mean(0)).detach().cpu().numpy().tolist()
        labels = y.detach().cpu().numpy().tolist()
        all_probs += probs
        all_labels += labels

labels_np, probs_np = np.array(all_labels), np.array(all_probs)
    acc = float(((probs_np >= 0.5).astype(int) == labels_np).mean())
    auc = float(roc_auc_score(labels_np, probs_np)) if len(np.unique(labels_np))>1 else float("nan")
    return {"labels": all_labels, "probs": all_probs, "AUC": auc, "ACC": acc}

# (misc)
def collect_eval_cache(loader, attack_names=None, tta_n=1):
    if attack_names is None:
        attack_names = list(ATTACK_FUNCS.keys())
    cache = {}
    cache["Clean"] = _eval_with_labels(loader, None, tta_n=tta_n)
    for i,a in enumerate(attack_names):
        cache[a] = _eval_with_labels(loader, a, tta_n=1)  # TTA=1 to keep runtime lighter
    return cache

def pick_worst_attack(eval_cache, by="ACC"):
    """by ∈ {'ACC','AUC','ΔAUC'}; returns attack name."""
    clean_auc = eval_cache["Clean"]["AUC"]
    worst_name, worst_val = None, +1e9
    for k,v in eval_cache.items():
        if k == "Clean": continue
        if by == "ACC":
            val = v["ACC"]
        elif by == "AUC":
            val = v["AUC"]
        else:  # ΔAUC magnitude (most drop)
            val = clean_auc - v["AUC"]
        if by in ("ACC","AUC"):
            # lower is worse
            if val < worst_val: worst_val, worst_name = val, k
        else:
            # higher drop is worse
            if val > worst_val: worst_val, worst_name = val, k
    return worst_name

# ΔAUC bars
def plot_delta_auc_and_acc(eval_cache):
    clean_auc = eval_cache["Clean"]["AUC"]
    clean_acc = eval_cache["Clean"]["ACC"]
    attacks = [k for k in eval_cache.keys() if k != "Clean"]
    deltas  = [eval_cache[a]["AUC"] - clean_auc for a in attacks]
    accs    = [eval_cache[a]["ACC"] for a in attacks]

# (misc)
fig, ax1 = plt.subplots(figsize=(1.6*max(6,len(attacks)), 4.5))
    x = np.arange(len(attacks))
    colors = [_color_for(a,i) for i,a in enumerate(attacks)]

bars = ax1.bar(x, deltas, color=colors, alpha=0.9)
    ax1.axhline(0, color="#444", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels(attacks, rotation=25, ha="right")
    ax1.set_ylabel("ΔAUC vs Clean")
    ax1.set_title("Robustness by Attack: ΔAUC (bars) + ACC (dots)")

# (misc)
ax2 = ax1.twinx()
    ax2.plot(x, accs, "o-", lw=2, alpha=0.9, color="#111111")
    ax2.axhline(clean_acc, ls="--", lw=1, color="#666666", label=f"Clean ACC={clean_acc:.3f}")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("ACC")
    ax2.legend(loc="lower right")
    plt.tight_layout(); plt.show()

# ROC overlay (Clean vs Worst)
def plot_roc_clean_vs_worst(eval_cache, by="ACC"):
    worst = pick_worst_attack(eval_cache, by=by)
    y_c, p_c = np.array(eval_cache["Clean"]["labels"]).astype(int), np.array(eval_cache["Clean"]["probs"]).astype(float)
    y_w, p_w = np.array(eval_cache[worst]["labels"]).astype(int), np.array(eval_cache[worst]["probs"]).astype(float)

fpr_c, tpr_c, _ = roc_curve(y_c, p_c)
    fpr_w, tpr_w, _ = roc_curve(y_w, p_w)

# (misc)
def tpr_at_fpr(fpr, tpr, target):
        ok = np.where(fpr <= target)[0]
        return tpr[ok].max() if len(ok)>0 else np.nan

plt.figure(figsize=(6,5))
    plt.semilogx(fpr_c, tpr_c, lw=2, label=f"Clean (AUC={eval_cache['Clean']['AUC']:.3f})")
    plt.semilogx(fpr_w, tpr_w, lw=2, label=f"{worst} (AUC={eval_cache[worst]['AUC']:.3f})")

# (misc)
for target in [1e-2, 1e-3]:
        tc = tpr_at_fpr(fpr_c, tpr_c, target)
        tw = tpr_at_fpr(fpr_w, tpr_w, target)
        plt.scatter([target, target], [tc, tw], marker="o", s=50, zorder=5)
        plt.text(target*1.1, tc, f"{tc:.2f}", va="bottom", fontsize=9)
        plt.text(target*1.1, tw, f"{tw:.2f}", va="bottom", fontsize=9)

plt.xlim(1e-4, 1.0); plt.ylim(0.0, 1.0)
    plt.xlabel("FPR (log)"); plt.ylabel("TPR")
    plt.title(f"ROC — Clean vs Worst ({worst})")
    plt.grid(True, which="both", alpha=0.3); plt.legend(loc="lower right")
    plt.show()

# (misc)
# DET overlay (FNR vs FPR)
def plot_det_clean_vs_worst(eval_cache, by="ACC"):
    worst = pick_worst_attack(eval_cache, by=by)
    y_c, p_c = np.array(eval_cache["Clean"]["labels"]).astype(int), np.array(eval_cache["Clean"]["probs"]).astype(float)
    y_w, p_w = np.array(eval_cache[worst]["labels"]).astype(int), np.array(eval_cache[worst]["probs"]).astype(float)

fpr_c, tpr_c, _ = roc_curve(y_c, p_c); fnr_c = 1.0 - tpr_c
    fpr_w, tpr_w, _ = roc_curve(y_w, p_w); fnr_w = 1.0 - tpr_w

# (misc)
plt.figure(figsize=(6,5))
    plt.semilogx(fpr_c, fnr_c, lw=2, label="Clean")
    plt.semilogx(fpr_w, fnr_w, lw=2, label=f"Worst: {worst}")
    plt.xlabel("FPR (log)"); plt.ylabel("FNR")
    plt.title("DET — Clean vs Worst")
    plt.grid(True, which="both", alpha=0.3); plt.legend(loc="upper right")
    plt.xlim(1e-4, 1.0); plt.ylim(0.0, 1.0)
    plt.show()

# (misc)
# Reliability overlay (Clean vs Worst)
def plot_calibration_overlay(eval_cache, by="ACC", n_bins=12):
    worst = pick_worst_attack(eval_cache, by=by)
    def _curve(name):
        y = np.array(eval_cache[name]["labels"]).astype(int)
        p = np.array(eval_cache[name]["probs"]).astype(float)
        yhat = (p>=0.5).astype(int)
        conf = np.where(yhat==1, p, 1.0-p)
        qs = np.quantile(conf, np.linspace(0,1,n_bins+1)); qs[0], qs[-1] = 0.0, 1.0
        centers, accs = [], []
        for i in range(n_bins):
            m = (conf >= qs[i]) & (conf <= qs[i+1] if i==n_bins-1 else conf < qs[i+1])
            if m.sum()==0: continue
            centers.append(conf[m].mean()); accs.append((yhat[m]==y[m]).mean())
        return np.array(centers), np.array(accs)
    xc, yc = _curve("Clean")
    xw, yw = _curve(worst)

plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.plot(xc, yc, "o-", lw=2, label=f"Clean (ECE {ece_binary(eval_cache['Clean']['labels'], eval_cache['Clean']['probs']):.3f})")
    plt.plot(xw, yw, "o-", lw=2, label=f"{worst} (ECE {ece_binary(eval_cache[worst]['labels'], eval_cache[worst]['probs'])::.3f})")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability (equal-mass bins)")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    plt.show()

# (misc)
# Risk–Coverage overlay
def risk_coverage(labels, probs):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat==1, p, 1.0 - p)
    order = np.argsort(-conf)
    y_ord, yhat_ord, conf_ord = y[order], yhat[order], conf[order]
    cov, risk = [], []
    correct = 0
    for k in range(1, len(y_ord)+1):
        correct += int(yhat_ord[k-1] == y_ord[k-1])
        cov.append(k / len(y_ord))
        risk.append(1.0 - correct / k)
    cov, risk = np.array(cov), np.array(risk)
    aurc = float(np.trapz(risk, cov))
    return {"AURC": aurc, "coverage": cov, "risk": risk}

# (misc)
def plot_risk_coverage_overlay(eval_cache, by="ACC"):
    worst = pick_worst_attack(eval_cache, by=by)
    rc_clean = risk_coverage(eval_cache["Clean"]["labels"], eval_cache["Clean"]["probs"])
    rc_worst = risk_coverage(eval_cache[worst]["labels"], eval_cache[worst]["probs"])
    plt.figure(figsize=(6,5))
    plt.plot(rc_clean["coverage"], rc_clean["risk"], lw=2, label=f"Clean (AURC={rc_clean['AURC']:.4f})")
    plt.plot(rc_worst["coverage"], rc_worst["risk"], lw=2, label=f"{worst} (AURC={rc_worst['AURC']:.4f})")
    plt.xlabel("Coverage"); plt.ylabel("Risk (error rate)")
    plt.title("Risk–Coverage (Selective prediction)")
    plt.grid(True, alpha=0.3); plt.ylim(0,1); plt.xlim(0,1); plt.legend()
    plt.show()

# (misc)
# 1) Build eval cache once (uses your test split)
attack_names = list(ATTACK_FUNCS.keys())
ec = collect_eval_cache(test_loader, attack_names=attack_names, tta_n=TTA_N)  # TTA for Clean only

# 2) ΔAUC + ACC
plot_delta_auc_and_acc(ec)

# 3) ROC / DET overlays (Clean vs Worst attack)
plot_roc_clean_vs_worst(ec, by="ACC")   # or by="ΔAUC"

# (misc)
plot_det_clean_vs_worst(ec, by="ACC")

# (misc)
# 4) Calibration overlay
plot_calibration_overlay(ec, by="ACC", n_bins=12)

# (misc)
# 5) Risk–Coverage (abstention) overlay
plot_risk_coverage_overlay(ec, by="ACC")

# One-shot table for Clean + all Attacks (detection, calibration, abstention, weak-loc)
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from PIL import Image

@torch.no_grad()
def _eval_with_labels(loader, attack_name=None, tta_n=1):
    """
    Returns dict with labels/probs and base metrics (AUC/AP/ACC/ECE) for Clean (attack_name=None)
    or a specific attack (jpeg/warp/...).
    - Mean over TTA for logits.
    """
    model.eval()
    all_labels, all_probs = [], []

# (misc)
for x, m, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits_list = []
        for t in range(max(1, int(tta_n))):
            if t == 0:
                if attack_name is None:
                    xx = x if x.shape[-1] == IMG_SIZE else F.interpolate(x, (IMG_SIZE, IMG_SIZE))
                else:
                    pils = denorm_to_pil(x)
                    ims = [ATTACK_FUNCS[attack_name](im) for im in pils]
                    xx = norm_batch(tensorize_pil_list(ims, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
                logit, _ = model(xx)
            else:
                pils = denorm_to_pil(x)
                jitter = []
                for im in pils:
                    w, h = im.size; dw, dh = int(0.02*w), int(0.02*h)
                    left = np.random.randint(0, max(1,dw)+1); top = np.random.randint(0, max(1,dh)+1)
                    im2 = im.crop((left, top, w-left, h-top)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                    jitter.append(im2)
                xx = norm_batch(tensorize_pil_list(jitter, resize_to=(IMG_SIZE, IMG_SIZE)).to(device))
                logit, _ = model(xx)
            logits_list.append(logit)

probs = torch.sigmoid(torch.stack(logits_list, 0).mean(0)).detach().cpu().numpy().tolist()
        labels = y.detach().cpu().numpy().tolist()
        all_probs += probs; all_labels += labels

y = np.array(all_labels).astype(int); p = np.array(all_probs).astype(float)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap  = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    acc = float(((p >= 0.5).astype(int) == y).mean())
    ece = float(ece_binary(y, p, n_bins=15))
    return {"labels": all_labels, "probs": all_probs, "AUC": auc, "AP": ap, "ACC": acc, "ECE": ece}

# (misc)
def risk_coverage(labels, probs):
    """AURC for selective prediction (lower is better)."""
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat==1, p, 1.0 - p)
    order = np.argsort(-conf)
    y_ord, yhat_ord = y[order], yhat[order]
    cov, risk = [], []; correct = 0
    for k in range(1, len(y_ord)+1):
        correct += int(yhat_ord[k-1] == y_ord[k-1])
        cov.append(k/len(y_ord)); risk.append(1.0 - correct/k)
    return float(np.trapz(risk, cov))

# (misc)
def _maybe(val):
    try:
        return float(val)
    except Exception:
        return val

def build_full_metrics_table(loader, attack_names=None, tau=0.5, tta_n_clean=3,
                             include_weakloc=True, weakloc_thr=0.3, include_iou=False):
    """
    Returns a pandas DataFrame with rows: Clean + each attack.
    Columns: AUC, AP, ACC, ECE, EER, tau@EER, APCER/BPCER/HTER (0.5 & EER),
             TPR@FPR, F1, MCC, Brier, NLL, AURC, EWR, Precision-in-ROI@thr,
             (optional) IoU@0.3, IoU_soft.
    """
    if attack_names is None:
        attack_names = list(ATTACK_FUNCS.keys())

# (misc)
# Clean
    cache = {}
    cache["Clean"] = _eval_with_labels(loader, None, tta_n=tta_n_clean)
    for a in attack_names:
        cache[a] = _eval_with_labels(loader, a, tta_n=1)  # faster

rows = []
    for split in ["Clean"] + attack_names:
        y, p = cache[split]["labels"], cache[split]["probs"]
        base = {k: cache[split][k] for k in ["AUC","AP","ACC","ECE"]}

# (misc)
# Extra detection metrics
        extra = compute_extra_metrics(y, p, tau=tau)  # from earlier cell

# (misc)
# Abstention
        aurc = risk_coverage(y, p)

# (misc)
row = {"Split": split, **base, **extra, "AURC": aurc}

# (misc)
# IoU (optional; slower, esp. for attacks)
        if include_iou:
            # reuse weak_loc_eval’s Dilated_IoU as proxy OR compute precise IoU if you kept that helper
            row["IoU@0.3"]  = wl.get(f"Dilated_IoU@{0.3:.2f}(r=12)", np.nan)  # tolerant IoU
            row["IoU_soft"] = np.nan  # or plug your soft-IoU if you have it cached

# (misc)
rows.append({k: _maybe(v) for k,v in row.items()})

# Column order
    col_order = [
        "Split",
        "AUC","AP","ACC","ECE",
        "EER","tau@EER",
        "APCER@0.5","BPCER@0.5","HTER@0.5",
        "APCER@EER","BPCER@EER","HTER@EER",
        "TPR@FPR=1e-2","TPR@FPR=1e-3",
        "F1@0.5","MCC@0.5",
        "Brier","NLL",
        "AURC",
        "IoU@0.3","IoU_soft"
    ]
    df = pd.DataFrame(rows)
    df = df[[c for c in col_order if c in df.columns]]

# (misc)
# Round floats for readability
    for c in df.columns:
        if df[c].dtype.kind in "f":
            df[c] = df[c].round(4)
    return df

# (misc)
# ===== Run it =====
attacks = list(ATTACK_FUNCS.keys())
full_table = build_full_metrics_table(
    test_loader, attack_names=attacks, tau=0.5, tta_n_clean=3,
    include_weakloc=True, weakloc_thr=0.3, include_iou=False  # set True to add IoU columns (slower)
)
full_table

# 🔧 Find a single global threshold that maximizes worst-case ACC across all attacks
import numpy as np
from sklearn.metrics import confusion_matrix

# (misc)
def _acc_at_threshold(labels, probs, tau):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    yhat = (p >= tau).astype(int)
    return float((yhat == y).mean())

# (misc)
def _grid_from_cache(eval_cache, n=1001):
    # Use all predicted probabilities seen to make a smart grid
    vals = []
    for k,v in eval_cache.items():
        vals.extend(v["probs"])
    vals = np.array(vals, dtype=float)
    # include 0 and 1 explicitly; dense around observed scores
    grid = np.unique(np.concatenate([
        np.linspace(0,1,n),
        vals, np.nextafter(vals, 0), np.nextafter(vals, 1)
    ]))
    grid = grid[(grid>=0.0) & (grid<=1.0)]
    return np.sort(grid)

# (misc)
def find_global_tau_star(eval_cache, grid=None):
    if grid is None:
        grid = _grid_from_cache(eval_cache, n=1001)
    splits = list(eval_cache.keys())
    best_tau, best_min_acc = 0.5, -1.0
    per_tau = []
    for tau in grid:
        accs = {s: _acc_at_threshold(eval_cache[s]["labels"], eval_cache[s]["probs"], tau) for s in splits}
        worst = min(accs.values())
        per_tau.append((tau, worst, accs))
        if worst > best_min_acc:
            best_min_acc, best_tau = worst, tau
    return best_tau, best_min_acc, per_tau

def report_at_tau(eval_cache, tau, show_confusion=False):
    print(f"\n=== Report at global τ* = {tau:.4f} ===")
    rows = []
    for k,v in eval_cache.items():
        y = np.array(v["labels"]).astype(int)
        p = np.array(v["probs"]).astype(float)
        yhat = (p >= tau).astype(int)
        acc = float((yhat==y).mean())
        rows.append((k, acc, y.sum(), (1-y).sum()))
        if show_confusion:
            tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
            print(f"{k:12s} ACC={acc:.4f} | TN={tn} FP={fp} FN={fn} TP={tp}")
    # neat table
    rows_sorted = sorted(rows, key=lambda r: (r[0]!="Clean", r[0]))  # Clean first
    print("\nSplit\t\tACC@τ*\t#Fake\t#Real")
    for s,acc,nf,nr in rows_sorted:
        print(f"{s:12s}\t{acc:.4f}\t{nf}\t{nr}")

# (misc)
# 1) Build eval cache (Clean uses your TTA_N; attacks use TTA=1 to save time)
attacks = list(ATTACK_FUNCS.keys())
ec = collect_eval_cache(test_loader, attack_names=attacks, tta_n=TTA_N)

# (misc)
# 2) Find global robust threshold
tau_star, worst_acc, _ = find_global_tau_star(ec)
print(f"Global τ* = {tau_star:.4f} (maximizes worst-case ACC={worst_acc:.4f})")

# (misc)
# 3) Report per-split performance at τ*
report_at_tau(ec, tau_star, show_confusion=True)

# (misc)
# Put this once (e.g., above your metric helpers)
import numpy as np
from sklearn.metrics import log_loss as _sk_log_loss

# (misc)
def safe_log_loss(y_true, p_pos, labels=[0,1], clip=1e-7):
    p_pos = np.asarray(p_pos, dtype=float)
    p_pos = np.clip(p_pos, clip, 1.0 - clip)
    return float(_sk_log_loss(np.asarray(y_true).astype(int), p_pos, labels=labels))

# (misc)
# 🧾 Metrics table at a fixed global threshold τ*
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, log_loss

# (misc)
def _metrics_at_tau(labels, probs, tau):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)

# (misc)
yhat = (p >= tau).astype(int)
    acc   = float((yhat == y).mean())
    apcer = float(((yhat == 0) & (y == 1)).mean())  # attacks accepted as real
    bpcer = float(((yhat == 1) & (y == 0)).mean())  # reals rejected as fake
    hter  = 0.5*(apcer + bpcer)
    f1    = float(f1_score(y, yhat))
    mcc   = float(matthews_corrcoef(y, yhat))

# Proper scoring rules (threshold-free)
    brier = float(np.mean((p - y)**2))
    nll   = safe_log_loss(y, p, labels=[0,1])  # <-- no eps arg; uses clipping

return {
        "ACC@τ*": acc, "APCER@τ*": apcer, "BPCER@τ*": bpcer, "HTER@τ*": hter,
        "F1@τ*": f1, "MCC@τ*": mcc, "Brier": brier, "NLL": nll,
    }

# (misc)
def _aurc(labels, probs):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    yhat = (p >= 0.5).astype(int)                      # confidence uses predicted class
    conf = np.where(yhat==1, p, 1.0 - p)
    order = np.argsort(-conf)
    y_ord, yhat_ord = y[order], yhat[order]
    cov, risk, correct = [], [], 0
    for k in range(1, len(y_ord)+1):
        correct += int(yhat_ord[k-1] == y_ord[k-1])
        cov.append(k/len(y_ord)); risk.append(1.0 - correct/k)
    return float(np.trapz(risk, cov))

def build_table_at_tau(eval_cache, tau_star, include_base=True):
    """
    eval_cache: dict from collect_eval_cache(...), with keys 'Clean' + attacks,
                and values containing 'labels' and 'probs' (lists).
    tau_star  : global threshold to apply (e.g., 0.8572)
    include_base: if True, also include AUC/AP/ECE from eval_cache dicts.
    """
    rows = []
    keys = ["Clean"] + [k for k in eval_cache.keys() if k != "Clean"]
    for k in keys:
        y = eval_cache[k]["labels"]; p = eval_cache[k]["probs"]
        row = {"Split": k, **_metrics_at_tau(y, p, tau_star)}
        # carry over threshold-free metrics if present
        if include_base:
            for base in ["AUC","AP","ECE"]:
                if base in eval_cache[k]:
                    row[base] = float(eval_cache[k][base])
        row["AURC"] = _aurc(y, p)
        rows.append(row)
    df = pd.DataFrame(rows)

# nice column order
    order = ["Split",
             "AUC","AP","ECE",
             "ACC@τ*","APCER@τ*","BPCER@τ*","HTER@τ*",
             "F1@τ*","MCC@τ*","Brier","NLL","AURC"]
    df = df[[c for c in order if c in df.columns]]
    # round floats
    for c in df.columns:
        if df[c].dtype.kind in "f":
            df[c] = df[c].round(4)
    return df

# (misc)
tau_star = 0.8572  # your τ*
table_tau = build_table_at_tau(ec, tau_star, include_base=True)
table_tau

from sklearn.metrics import confusion_matrix

def confusion_summary(eval_cache, tau_star):
    rows=[]
    for k,v in eval_cache.items():
        y = np.array(v["labels"]).astype(int)
        p = np.array(v["probs"]).astype(float)
        yhat = (p >= tau_star).astype(int)
        tn,fp,fn,tp = confusion_matrix(y,yhat,labels=[0,1]).ravel()
        rows.append([k,int(tp),int(tn),int(fp),int(fn)])
    return pd.DataFrame(rows, columns=["Split","TP","TN","FP","FN"]).sort_values("Split")

# (misc)
confusion_summary(ec, 0.8572)

# Bootstrap CIs for ACC@τ* and ECE
import numpy as np, pandas as pd
from tqdm import trange

def bootstrap_ci_acc_ece(labels, probs, tau, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    n_samples = len(y)
    accs, eces = [], []
    for _ in trange(n, leave=False):
        idx = rng.integers(0, n_samples, size=n_samples)
        yy, pp = y[idx], p[idx]
        yhat = (pp >= tau).astype(int)
        accs.append((yhat == yy).mean())
        # predicted-class confidence ECE (15 bins)
        conf = np.where(yhat==1, pp, 1.0-pp)
        bins = np.linspace(0,1,16)
        ece = 0.0
        for i in range(15):
            m = (conf >= bins[i]) & (conf < (bins[i+1]) if i<14 else conf <= bins[i+1])
            if m.sum()==0: continue
            acc_bin = (yhat[m]==yy[m]).mean()
            ece += m.mean()*abs(acc_bin - conf[m].mean())
        eces.append(ece)
    lo_acc, hi_acc = np.percentile(accs, [2.5,97.5])
    lo_ece, hi_ece = np.percentile(eces, [2.5,97.5])
    return (lo_acc, hi_acc), (lo_ece, hi_ece)

# Example: add CI columns to your table at τ*
def attach_ci_columns(eval_cache, tau_star, df):
    splits = df["Split"].tolist()
    acc_lo, acc_hi, ece_lo, ece_hi = [], [], [], []
    for s in splits:
        y, p = eval_cache[s]["labels"], eval_cache[s]["probs"]
        (loA, hiA), (loE, hiE) = bootstrap_ci_acc_ece(y, p, tau_star, n=1000)
        acc_lo.append(round(loA,4)); acc_hi.append(round(hiA,4))
        ece_lo.append(round(loE,4)); ece_hi.append(round(hiE,4))
    df = df.copy()
    df["ACC@τ* [2.5%,97.5%]"] = [f"[{a:.4f},{b:.4f}]" for a,b in zip(acc_lo, acc_hi)]
    df["ECE [2.5%,97.5%]"]    = [f"[{a:.4f},{b:.4f}]" for a,b in zip(ece_lo, ece_hi)]
    return df

# (misc)
# Usage:
table_tau = build_table_at_tau(ec, tau_star, include_base=True)

# (misc)
table_tau_ci = attach_ci_columns(ec, tau_star, table_tau); table_tau_ci

# (misc)
# Score histograms: Clean vs Regrain (reals/fakes separated)
import numpy as np, matplotlib.pyplot as plt

# (misc)
def plot_score_hists(eval_cache, attack="regrain"):
    def split_scores(name):
        y = np.array(eval_cache[name]["labels"]).astype(int)
        p = np.array(eval_cache[name]["probs"]).astype(float)
        return p[y==0], p[y==1]  # reals, fakes
    cr, cf = split_scores("Clean")
    rr, rf = split_scores(attack)

# (misc)
bins = np.linspace(0,1,41)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.hist(cr, bins, alpha=0.5, label="Clean REAL"); plt.hist(rr, bins, alpha=0.5, label="Regrain REAL")
    plt.title("Scores for REAL"); plt.xlabel("p(fake)"); plt.ylabel("count"); plt.legend()
    plt.subplot(1,2,2); plt.hist(cf, bins, alpha=0.5, label="Clean FAKE"); plt.hist(rf, bins, alpha=0.5, label="Regrain FAKE")
    plt.title("Scores for FAKE"); plt.xlabel("p(fake)"); plt.legend()
    plt.tight_layout(); plt.show()

# (misc)
# Usage (ec from collect_eval_cache(...)):
plot_score_hists(ec, attack="regrain")
