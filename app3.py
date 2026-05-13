"""
Marine Plastic Debris Detection System
Three-Stage Deep Learning Pipeline
Author: Hamdi Braiek
"""

import streamlit as st
import torch
import numpy as np
import cv2
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Plastic Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — FORCED DARK + HIGH CONTRAST
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg0:   #060e1a;
    --bg1:   #0c1a2e;
    --bg2:   #112240;
    --teal:  #00e5cc;
    --teal2: #00b39e;
    --blue:  #5badff;
    --purp:  #c084fc;
    --yell:  #fbbf24;
    --red:   #f87171;
    --grn:   #4ade80;
    --w0:    #f0f6ff;
    --w1:    #cbd5e1;
    --w2:    #94a3b8;
    --w3:    #475569;
    --bdr:   rgba(255,255,255,0.09);
    --bdrt:  rgba(0,229,204,0.30);
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.main, .main .block-container,
section.main {
    background-color: var(--bg0) !important;
    color: var(--w0) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: var(--bg1) !important;
    border-right: 1px solid var(--bdrt) !important;
}
[data-testid="stSidebar"] * {
    color: var(--w0) !important;
}

p, span, div, label, li, small {
    color: var(--w0) !important;
}
h1,h2,h3,h4,h5,h6 {
    font-family: 'Space Mono', monospace !important;
    color: var(--w0) !important;
}

/* Hero */
.hero {
    background: linear-gradient(
        135deg,
        rgba(0,229,204,0.11),
        rgba(91,173,255,0.09));
    border: 1px solid var(--bdrt);
    border-radius: 14px;
    padding: 1.7rem 2rem;
    margin-bottom: 1.5rem;
}
.hero-t {
    font-family:'Space Mono',monospace;
    font-size:1.7rem;font-weight:700;
    color:var(--teal) !important;margin:0;
}
.hero-s {
    font-size:0.92rem;
    color:var(--w1) !important;
    margin-top:0.4rem;
}

/* Stage cards */
.sc {
    border-radius:10px;
    padding:1rem 1.2rem;
    margin-bottom:0.7rem;
    border-left:3px solid;
}
.sc1{background:rgba(0,229,204,0.07);
     border-color:var(--teal);}
.sc2{background:rgba(91,173,255,0.07);
     border-color:var(--blue);}
.sc3{background:rgba(192,132,252,0.07);
     border-color:var(--purp);}
.sc-num{font-family:'Space Mono',monospace;
        font-size:0.62rem;letter-spacing:2px;
        text-transform:uppercase;
        color:var(--w2) !important;
        margin-bottom:0.2rem;}
.sc-title{font-size:0.93rem;font-weight:600;
          color:var(--w0) !important;}
.sc-model{font-family:'Space Mono',monospace;
          font-size:0.7rem;
          color:var(--w2) !important;
          margin-top:0.15rem;}

/* Metrics */
.mb{background:rgba(255,255,255,0.04);
    border:1px solid var(--bdr);
    border-radius:10px;padding:0.9rem 0.7rem;
    text-align:center;margin-bottom:0.5rem;}
.mb-v{font-family:'Space Mono',monospace;
      font-size:1.45rem;font-weight:700;
      line-height:1.1;}
.mb-l{font-size:0.7rem;
      color:var(--w1) !important;
      margin-top:0.25rem;line-height:1.4;}

/* Badges */
.bdg{display:inline-block;border-radius:20px;
     padding:0.28rem 0.9rem;
     font-family:'Space Mono',monospace;
     font-size:0.8rem;font-weight:700;
     margin-top:0.35rem;}
.bp{background:rgba(248,113,113,0.17);
    border:1.5px solid var(--red);
    color:#fca5a5 !important;}
.bc{background:rgba(74,222,128,0.14);
    border:1.5px solid var(--grn);
    color:#86efac !important;}
.bw{background:rgba(251,191,36,0.14);
    border:1.5px solid var(--yell);
    color:#fcd34d !important;}

/* Info box */
.ib{background:rgba(91,173,255,0.06);
    border:1px solid rgba(91,173,255,0.22);
    border-radius:9px;
    padding:0.85rem 1rem;
    font-size:0.83rem;
    color:var(--w1) !important;
    margin:0.6rem 0;line-height:1.6;}
.ib b{color:var(--teal) !important;}
.ib a{color:var(--blue) !important;}

/* Buttons */
.stButton>button{
    background:linear-gradient(
        135deg,var(--teal),var(--teal2)
    ) !important;
    color:#060e1a !important;
    font-family:'Space Mono',monospace !important;
    font-weight:700 !important;
    font-size:0.83rem !important;
    border:none !important;
    border-radius:8px !important;
    letter-spacing:1px;
}
.stButton>button:hover{opacity:0.85;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    background:rgba(255,255,255,0.04);
    border-radius:10px;padding:3px;gap:3px;
}
.stTabs [data-baseweb="tab"]{
    font-family:'Space Mono',monospace;
    font-size:0.76rem;
    color:var(--w1) !important;
    border-radius:7px;
}
.stTabs [aria-selected="true"]{
    background:rgba(0,229,204,0.14) !important;
    color:var(--teal) !important;
}

/* Progress */
.stProgress>div>div>div{
    background:linear-gradient(
        90deg,var(--teal),var(--blue)
    ) !important;
}

/* Upload */
[data-testid="stFileUploader"]{
    background:rgba(0,229,204,0.04) !important;
    border:2px dashed var(--bdrt) !important;
    border-radius:10px !important;
}

/* DataFrame */
[data-testid="stDataFrame"] *{
    color:var(--w0) !important;
    background:transparent !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio span,
[data-testid="stSidebar"] .stSlider label{
    color:var(--w1) !important;
}

hr{border-color:var(--bdr) !important;}

[data-testid="stDecoration"]{
    display:none !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
class MarineDebrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 6, kernel_size=5,
            padding="same")
        self.conv2 = nn.Conv2d(
            6, 16, kernel_size=5,
            padding="same")
        self.conv3 = nn.Conv2d(
            16, 64, kernel_size=3,
            padding="valid")
        self.conv4 = nn.Conv2d(
            64, 32, kernel_size=3,
            padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1  = nn.Linear(
            13 * 13 * 32, 128)
        self.fc2  = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MarineDebrisClassifierV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(
            weights=None)
        self.base.classifier[1] = (
            nn.Linear(1280, 2))

    def forward(self, x):
        return self.base(x)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DEVICE      = torch.device('cpu')
CLASS_NAMES = ['no-plastic', 'plastic']

EVAL_TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

COLORS_BGR = [
    (0, 229, 204), (91, 173, 255),
    (192, 132, 252), (251, 191, 36),
    (248, 113, 113), (74, 222, 128),
]


# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_mobilenet():
    for p in [
        "models/MobileNetV2_best.pth",
        "models/CustomCNN_best.pth"
    ]:
        if os.path.exists(p):
            try:
                m = MarineDebrisClassifierV2()
                m.load_state_dict(
                    torch.load(
                        p, map_location=DEVICE))
                m.eval()
                return m, None
            except Exception as e:
                return None, str(e)
    return None, "MobileNetV2 weights not found"


@st.cache_resource
def load_cnn():
    for p in [
        "models/CustomCNN_best.pth",
        "models/CustomCNN_best.pth"
    ]:
        if os.path.exists(p):
            try:
                m = MarineDebrisClassifier()
                m.load_state_dict(
                    torch.load(
                        p, map_location=DEVICE))
                m.eval()
                return m, None
            except Exception as e:
                return None, str(e)
    return None, "CustomCNN weights not found"


@st.cache_resource
def load_yolo():
    """Load YOLOv8n-seg model."""
    try:
        from ultralytics import YOLO
        weight_path = (
            "yolov8n_seg_v2/weights/best.pt")
        if not os.path.exists(weight_path):
            weight_path = "models/yolov8n_trained_final.pt"
        if not os.path.exists(weight_path):
            return None, "YOLO weights not found."
        model = YOLO(weight_path)
        return model, None
    except Exception as e:
        return None, str(e)



@st.cache_resource
def load_sam():
    try:
        from segment_anything import (
            sam_model_registry, SamPredictor)
        p = "sam_models/sam_vit_h_4b8939.pth"
        if not os.path.exists(p):
            return None, None, (
                "SAM weights not found")
        sam  = sam_model_registry["vit_h"](
            checkpoint=p)
        sam.to(DEVICE)
        pred = SamPredictor(sam)
        return sam, pred, None
    except Exception as e:
        return None, None, str(e)


# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────
def classify(pil_img, model):
    t0 = time.time()
    t  = EVAL_TR(pil_img).unsqueeze(0)
    with torch.no_grad():
        o = model(t)
        p = torch.softmax(o, dim=1)[0]
        i = torch.argmax(p).item()
    return {
        'label':      CLASS_NAMES[i],
        'is_plastic': i == 1,
        'conf':       p[i].item(),
        'probs':      p.cpu().numpy(),
        'ms':         (time.time()-t0)*1000
    }


def detect(pil_img, yolo, conf=0.25):
    t0  = time.time()
    arr = np.array(pil_img)
    res = yolo(arr, conf=conf,
               verbose=False)[0]

    boxes, labels, confs, ymasks = \
        [], [], [], []

    if res.boxes is not None:
        for i, b in enumerate(
                res.boxes.xyxy.cpu().numpy()):
            boxes.append(b.astype(int))
            cid = int(
                res.boxes.cls[i].item())
            labels.append(
                res.names.get(cid, 'debris'))
            confs.append(
                float(
                    res.boxes.conf[i].item()))

    if res.masks is not None:
        for md in res.masks.data:
            m = md.cpu().numpy()
            mr = cv2.resize(
                m.astype(np.float32),
                (arr.shape[1], arr.shape[0]))
            ymasks.append(mr > 0.5)

    return {
        'boxes':  boxes, 'labels': labels,
        'confs':  confs, 'ymasks': ymasks,
        'n':      len(boxes),
        'ms':     (time.time()-t0)*1000,
        'arr':    arr
    }


def segment_sam(pil_img, boxes, pred):
    t0  = time.time()
    arr = np.array(pil_img)
    rgb = cv2.cvtColor(arr,
                       cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(rgb,
                       cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)
    masks = []
    for box in boxes:
        bt = torch.tensor(
            [box], dtype=torch.float,
            device=DEVICE)
        tb = pred.transform.apply_boxes_torch(
            bt, rgb.shape[:2])
        with torch.no_grad():
            m, sc, _ = pred.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=tb,
                multimask_output=False)
        best = m[torch.argmax(sc)] \
            .squeeze().cpu().numpy()
        masks.append(best)
    return {
        'masks': masks, 'n': len(masks),
        'ms':    (time.time()-t0)*1000,
        'arr':   arr
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def draw_boxes(arr, det):
    vis = arr.copy()
    for i, (box, lbl, cf) in enumerate(
            zip(det['boxes'],
                det['labels'],
                det['confs'])):
        c = COLORS_BGR[i % len(COLORS_BGR)]
        x1, y1, x2, y2 = box
        cv2.rectangle(
            vis, (x1, y1), (x2, y2), c, 2)
        txt = f"{lbl} {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX,
            0.48, 1)
        cv2.rectangle(
            vis,
            (x1, max(0, y1-th-8)),
            (x1+tw+6, y1), c, -1)
        cv2.putText(
            vis, txt,
            (x1+3, max(th, y1-4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (10, 18, 35), 1,
            cv2.LINE_AA)
    return vis


def figure10_panel(arr, masks):
    """
    Side-by-side: left = image + overlays,
    right = white masks on black.
    Matches Figure 10 in the paper.
    """
    h, w = arr.shape[:2]
    left  = arr.copy()
    right = np.zeros((h, w, 3),
                     dtype=np.uint8)

    for i, mask in enumerate(masks):
        c  = COLORS_BGR[i % len(COLORS_BGR)]
        ov = np.zeros_like(left)
        ov[mask] = c
        left = cv2.addWeighted(
            left, 0.78, ov, 0.22, 0)
        ctrs, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            left, ctrs, -1, c, 2)
        right[mask] = (215, 215, 215)

    sep = np.zeros((h, 8, 3),
                   dtype=np.uint8)
    sep[:] = (30, 50, 80)
    return np.hstack([left, sep, right])


def conf_bar(cls_r, model_name):
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    fig.patch.set_facecolor('#060e1a')
    ax.set_facecolor('#0c1a2e')
    probs  = cls_r['probs']
    colors = ['#4ade80', '#f87171']
    bars   = ax.barh(
        CLASS_NAMES, probs,
        color=colors, height=0.44,
        edgecolor='none')
    for bar, p in zip(bars, probs):
        ax.text(
            min(p+0.03, 0.93),
            bar.get_y()+bar.get_height()/2,
            f'{p:.1%}',
            va='center', ha='left',
            color='#f0f6ff',
            fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.18)
    ax.set_title(
        f'Model: {model_name}',
        color='#f0f6ff', fontsize=9, pad=5)
    ax.tick_params(
        colors='#cbd5e1', labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(
        'Confidence',
        color='#94a3b8', fontsize=8)
    plt.tight_layout(pad=0.4)
    return fig


def pipeline_flow(res, active):
    fig, ax = plt.subplots(figsize=(11, 1.85))
    fig.patch.set_facecolor('#060e1a')
    ax.set_facecolor('#060e1a')
    ax.axis('off')

    nodes = [
        ("INPUT",           "#334155"),
        ("MobileNetV2\n/ Custom CNN",
         "#00e5cc"),
        ("YOLOv8n-seg\nDetector",
         "#5badff"),
        ("SAM ViT-H\nSegmenter",
         "#c084fc"),
        ("OUTPUT\nMasks",   "#334155"),
    ]
    sts = ["", "", "", "", ""]
    if res.get('cls'):
        c = res['cls']
        sts[1] = ("✓ PLASTIC\n"
                  if c['is_plastic']
                  else "✓ CLEAN\n") + \
                 f"{c['conf']:.1%}"
    if res.get('det'):
        d = res['det']
        sts[2] = f"✓ {d['n']} detected"
    if res.get('seg'):
        s = res['seg']
        sts[3] = f"✓ {s['n']} masks"

    xs = [0.06, 0.25, 0.45, 0.65, 0.84]
    for i, ((lbl, col),
            st_txt, x, act) in enumerate(
            zip(nodes, sts, xs, active)):
        alpha = "22" if act else "10"
        ec    = col if act else "#334155"
        rect  = mpatches.FancyBboxPatch(
            (x-0.08, 0.07), 0.145, 0.86,
            boxstyle="round,pad=0.015",
            linewidth=1.5 if act else 0.6,
            edgecolor=ec,
            facecolor=ec+alpha,
            transform=ax.transAxes,
            clip_on=False)
        ax.add_patch(rect)
        ax.text(
            x, 0.66, lbl,
            transform=ax.transAxes,
            ha='center', va='center',
            color='#f0f6ff' if act
            else '#475569',
            fontsize=7.5,
            fontweight='bold',
            multialignment='center',
            fontfamily='monospace')
        if st_txt and act:
            ax.text(
                x, 0.25, st_txt,
                transform=ax.transAxes,
                ha='center', va='center',
                color=col, fontsize=6.8,
                multialignment='center')
        if i < len(nodes)-1:
            ax.annotate(
                '',
                xy=(xs[i+1]-0.082, 0.5),
                xytext=(x+0.068, 0.5),
                xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(
                    arrowstyle='->',
                    color='#94a3b8',
                    lw=1.4))
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.3rem">
        <span style="font-family:'Space Mono',
              monospace;font-size:0.95rem;
              color:#00e5cc;font-weight:700">
            🌊 Pipeline Settings
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(
        "<b>Stage 1 — Classifier</b>",
        unsafe_allow_html=True)
    clf_choice = st.radio(
        "Model",
        ["MobileNetV2 (recommended)",
         "Custom CNN (baseline)"],
        index=0,
        label_visibility="collapsed")
    use_mob = "MobileNetV2" in clf_choice

    bypass = st.toggle(
        "Bypass gate — always run all stages",
        value=False,
        help="Run YOLO+SAM even on clean images")

    st.divider()
    st.markdown(
        "<b>Stage 2 — Detector</b>",
        unsafe_allow_html=True)
    conf_thr = st.slider(
        "Confidence threshold",
        0.10, 0.90, 0.25, 0.05)

    st.divider()
    st.markdown(
        "<b>Stage 3 — Segmentation</b>",
        unsafe_allow_html=True)
    seg_mode = st.radio(
        "Source",
        ["YOLO built-in masks",
         "SAM ViT-H (zero-shot, slow on CPU)"],
        index=0,
        label_visibility="collapsed")
    use_sam = "SAM" in seg_mode

    st.divider()
    st.markdown(
        "<span style='color:#94a3b8;"
        "font-size:0.75rem;font-weight:600'>"
        "MODEL STATUS</span>",
        unsafe_allow_html=True)

    mob_m, mob_e = load_mobilenet()
    cnn_m, cnn_e = load_cnn()
    yolo_m, y_e  = load_yolo()

    for nm, m, e in [
        ("MobileNetV2", mob_m, mob_e),
        ("Custom CNN",  cnn_m, cnn_e),
        ("YOLOv8n-seg", yolo_m, y_e)
    ]:
        ico = "🟢" if m else "🔴"
        msg = "ready" if m else "missing"
        st.markdown(
            f"<span style='font-size:0.78rem'>"
            f"{ico} <b>{nm}</b>: {msg}"
            f"</span>",
            unsafe_allow_html=True)

    if use_sam:
        _, sam_p, sam_e = load_sam()
        ico = "🟢" if sam_p else "🔴"
        msg = "ready" if sam_p else "missing"
        st.markdown(
            f"<span style='font-size:0.78rem'>"
            f"{ico} <b>SAM ViT-H</b>: {msg}"
            f"</span>",
            unsafe_allow_html=True)
    else:
        sam_p = None

    clf_m  = mob_m if use_mob else cnn_m
    clf_nm = ("MobileNetV2"
              if use_mob else "Custom CNN")
    clf_e  = mob_e if use_mob else cnn_e

    st.divider()
    st.markdown("""
    <div class="ib" style="font-size:0.68rem">
        <b>Paper:</b> Deep Learning-Based
        Framework for Plastic Debris Detection
        in Dynamic Aquatic Ecosystems<br>
        <b>Author:</b> Hamdi Braiek<br>
        <a href="https://github.com/braiekhamdi/marine-pollution-plastic">
            GitHub Repository
        </a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-t">
        🌊 Marine Plastic Detection System
    </div>
    <div class="hero-s">
        Three-stage deep learning pipeline —
        Classification → Detection →
        Segmentation
    </div>
</div>
""", unsafe_allow_html=True)

T1, T2, T3 = st.tabs([
    "🔬  Run Pipeline",
    "📊  Evaluation",
    "📖  About"
])


# ════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════
with T1:
    diag_ph = st.empty()
    with diag_ph:
        fig0 = pipeline_flow(
            {}, [True]*5)
        st.pyplot(
            fig0,
            use_container_width=True)
        plt.close()

    st.divider()
    st.markdown("#### Upload Image")
    up = st.file_uploader(
        "JPG / PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed")

    if not up:
        st.markdown("""
        <div class="ib">
            Upload an underwater image to
            run the full three-stage pipeline.
            Use the sidebar to configure
            models and options.
        </div>
        """, unsafe_allow_html=True)

    if up:
        pil = Image.open(up).convert("RGB")
        w, h = pil.size
        arr_orig = np.array(pil)

        ci, cx = st.columns([1, 1])
        with ci:
            st.image(
                pil,
                caption="Input image",
                use_container_width=True)
        with cx:
            st.markdown(f"""
            <div class="ib">
                <b>File:</b> {up.name}<br>
                <b>Size:</b> {w}×{h} px<br>
                <b>Classifier:</b>
                {clf_nm}<br>
                <b>Gate bypass:</b>
                {"ON" if bypass else "OFF"}<br>
                <b>Segmentation:</b>
                {"SAM ViT-H"
                 if use_sam
                 else "YOLO masks"}
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        go = st.button(
            "▶  RUN PIPELINE",
            use_container_width=True)

        if go:
            RES     = {}
            prog    = st.progress(0)
            active  = [True,True,
                       False,False,False]

            # ── STAGE 1 ──────────────────────
            s1_ph = st.status(
                "Stage 1 — Classification...",
                expanded=False)

            if clf_m is None:
                st.error(
                    f"Classifier not loaded:"
                    f" {clf_e}")
                st.stop()

            cls_r = classify(pil, clf_m)
            RES['cls'] = cls_r
            s1_ph.update(
                label=(
                    f"Stage 1 done — "
                    f"{cls_r['label'].upper()}"
                    f" ({cls_r['conf']:.1%})"),
                state="complete")
            prog.progress(25)
            active[2] = True

            with diag_ph:
                st.pyplot(
                    pipeline_flow(RES, active),
                    use_container_width=True)
                plt.close()

            # ── SHOW STAGE 1 ─────────────────
            st.markdown(
                "### Stage 1 — Classification")

            col1, col2 = st.columns([1, 1])
            with col1:
                is_p = cls_r['is_plastic']
                bclass = "bp" if is_p else "bc"
                btxt   = (
                    "🔴 PLASTIC DETECTED"
                    if is_p
                    else "🟢 CLEAN WATER")
                fwd = (
                    "✔ Forwarded to Stage 2"
                    if (is_p or bypass)
                    else
                    "✖ Pipeline stops here")
                fwd_c = (
                    "#4ade80"
                    if (is_p or bypass)
                    else "#f87171")

                st.markdown(f"""
                <div class="sc sc1">
                    <div class="sc-num">
                        STAGE 1 · {clf_nm}
                    </div>
                    <div class="sc-title">
                        Binary Classification
                    </div>
                    <div class="sc-model">
                        Confidence:
                        {cls_r['conf']:.1%}
                        · {cls_r['ms']:.0f} ms
                    </div>
                    <div style="margin-top:.45rem">
                        <span class="bdg {bclass}">
                            {btxt}
                        </span>
                    </div>
                    <div style="font-size:.76rem;
                                color:{fwd_c};
                                margin-top:.4rem;
                                font-weight:600">
                        {fwd}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                fig_cb = conf_bar(
                    cls_r, clf_nm)
                st.pyplot(
                    fig_cb,
                    use_container_width=True)
                plt.close()

            # ── GATE ─────────────────────────
            proceed = cls_r['is_plastic'] \
                      or bypass

            if not proceed:
                st.markdown("""
                <div class="ib">
                    ⏹️ <b>Pipeline stopped after
                    Stage 1.</b>
                    Image classified as clean
                    water. Enable
                    <b>Bypass gate</b> in the
                    sidebar to run all stages
                    regardless.
                </div>
                """, unsafe_allow_html=True)
                prog.progress(100)

            else:
                if bypass and not is_p:
                    st.markdown("""
                    <div class="ib">
                        ⚡ <b>Gate bypassed.</b>
                        Running YOLO + SAM on
                        this clean-water image.
                    </div>
                    """,
                    unsafe_allow_html=True)

                # ── STAGE 2 ──────────────────
                s2_ph = st.status(
                    "Stage 2 — YOLO Detection...",
                    expanded=False)

                if yolo_m is None:
                    st.error(
                        f"YOLO not loaded: {y_e}")
                    st.stop()

                det_r = detect(
                    pil, yolo_m,
                    conf=conf_thr)
                RES['det'] = det_r
                s2_ph.update(
                    label=(
                        f"Stage 2 done — "
                        f"{det_r['n']} "
                        f"object(s)"),
                    state="complete")
                prog.progress(60)
                active[3] = True

                with diag_ph:
                    st.pyplot(
                        pipeline_flow(
                            RES, active),
                        use_container_width=True)
                    plt.close()

                # ── SHOW STAGE 2 ─────────────
                st.markdown(
                    "### Stage 2 — Detection")
                d1, d2 = st.columns([1, 1])
                with d1:
                    st.markdown(f"""
                    <div class="sc sc2">
                        <div class="sc-num">
                            STAGE 2 · YOLOv8n-seg
                        </div>
                        <div class="sc-title">
                            Object Detection
                        </div>
                        <div class="sc-model">
                            {det_r['n']}
                            object(s) detected
                            · {det_r['ms']:.0f} ms
                        </div>
                    """, unsafe_allow_html=True)

                    if det_r['n'] > 0:
                        for i, (lb, cf) in \
                                enumerate(zip(
                                    det_r['labels'],
                                    det_r['confs'])):
                            bc  = COLORS_BGR[
                                i % len(COLORS_BGR)]
                            hex = (f"#{bc[2]:02x}"
                                   f"{bc[1]:02x}"
                                   f"{bc[0]:02x}")
                            st.markdown(
                                f"<div style='"
                                f"font-size:.79rem;"
                                f"margin-top:.25rem;"
                                f"color:{hex}'>"
                                f"▶ {lb} ({cf:.1%})"
                                f"</div>",
                                unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<span class="bdg bw">'
                            '⚠ No detections'
                            '</span>',
                            unsafe_allow_html=True)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True)

                with d2:
                    vis_det = draw_boxes(
                        det_r['arr'], det_r)
                    st.image(
                        vis_det,
                        caption=(
                            "Detected objects "
                            "with bounding boxes"),
                        use_container_width=True)

                prog.progress(75)

                # ── STAGE 3 ──────────────────
                st.markdown(
                    "### Stage 3 — Segmentation")

                if use_sam:
                    if sam_p is None:
                        st.markdown("""
                        <div class="ib">
                            ⚠ SAM not available.
                            Falling back to YOLO
                            built-in masks.
                        </div>
                        """,
                        unsafe_allow_html=True)
                        final_masks = (
                            det_r['ymasks'])
                        seg_nm = "YOLO masks"
                        seg_ms = 0
                    elif det_r['n'] == 0:
                        st.markdown("""
                        <div class="ib">
                            ⚠ No YOLO detections —
                            no prompts for SAM.
                        </div>
                        """,
                        unsafe_allow_html=True)
                        final_masks = []
                        seg_nm = "SAM ViT-H"
                        seg_ms = 0
                    else:
                        s3_ph = st.status(
                            "Stage 3 — SAM...",
                            expanded=False)
                        sam_r = segment_sam(
                            pil,
                            det_r['boxes'],
                            sam_p)
                        RES['seg'] = sam_r
                        final_masks = (
                            sam_r['masks'])
                        seg_nm = "SAM ViT-H"
                        seg_ms = sam_r['ms']
                        s3_ph.update(
                            label=(
                                f"Stage 3 done "
                                f"— {sam_r['n']}"
                                f" masks"),
                            state="complete")
                else:
                    final_masks = (
                        det_r['ymasks'])
                    seg_nm = "YOLO masks"
                    seg_ms = 0
                    RES['seg'] = {
                        'masks':   final_masks,
                        'n':       len(final_masks),
                        'ms':      0
                    }

                n_m = len(final_masks)
                active[4] = True

                with diag_ph:
                    st.pyplot(
                        pipeline_flow(
                            RES, active),
                        use_container_width=True)
                    plt.close()

                # Stage 3 info card
                st.markdown(f"""
                <div class="sc sc3">
                    <div class="sc-num">
                        STAGE 3 · {seg_nm}
                    </div>
                    <div class="sc-title">
                        Instance Segmentation
                    </div>
                    <div class="sc-model">
                        {n_m} mask(s) generated
                        · {seg_ms:.0f} ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if n_m > 0:
                    # ── Figure 10 Panel ───────
                    st.markdown(
                        "#### Segmentation "
                        "Output")
                    st.markdown("""
                    <div class="ib">
                        <b>Left:</b> original image
                        with colored overlays and
                        contour outlines per mask.
                        &nbsp;·&nbsp;
                        <b>Right:</b> binary mask
                        on black background —
                        white = detected plastic
                        debris.
                    </div>
                    """,
                    unsafe_allow_html=True)

                    panel = figure10_panel(
                        det_r['arr'],
                        final_masks)
                    st.image(
                        panel,
                        caption=(
                            "Left: segmented image"
                            "   |   "
                            "Right: binary mask"),
                        use_container_width=True)

                    # Overlay only
                    with st.expander(
                            "View overlay only"):
                        ov_img = (
                            det_r['arr'].copy())
                        for i, mk in enumerate(
                                final_masks):
                            c  = COLORS_BGR[
                                i % len(COLORS_BGR)]
                            ov = np.zeros_like(
                                ov_img)
                            ov[mk] = c
                            ov_img = cv2.addWeighted(
                                ov_img, 0.75,
                                ov, 0.25, 0)
                            ct, _ = cv2.findContours(
                                mk.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(
                                ov_img, ct,
                                -1, c, 2)
                        st.image(
                            ov_img,
                            caption="Overlay",
                            use_container_width=True)

                    # Individual masks
                    with st.expander(
                            f"Individual masks"
                            f" ({n_m})"):
                        cols = st.columns(
                            min(n_m, 4))
                        h2, w2 = (
                            det_r['arr'].shape[:2])
                        for i, mk in enumerate(
                                final_masks):
                            with cols[
                                    i % min(n_m, 4)]:
                                mv = np.zeros(
                                    (h2, w2, 3),
                                    dtype=np.uint8)
                                c  = COLORS_BGR[
                                    i % len(
                                        COLORS_BGR)]
                                mv[mk] = c
                                lb = (
                                    det_r['labels'][i]
                                    if i < len(
                                        det_r[
                                            'labels'])
                                    else
                                    f"obj {i+1}")
                                st.image(
                                    mv,
                                    caption=lb,
                                    use_container_width=True)
                else:
                    st.markdown("""
                    <div class="ib">
                        No masks to display.
                        Try lowering the confidence
                        threshold.
                    </div>
                    """,
                    unsafe_allow_html=True)

                prog.progress(100)

                # ── Timing ───────────────────
                st.divider()
                st.markdown(
                    "#### Processing Times")
                t_cls = cls_r['ms']
                t_det = det_r['ms']
                t_seg = seg_ms
                t_tot = t_cls + t_det + t_seg

                for col, (lbl, val, color) in \
                        zip(
                    st.columns(4),
                    [
                        ("Classification",
                         t_cls, "#00e5cc"),
                        ("Detection",
                         t_det, "#5badff"),
                        ("Segmentation",
                         t_seg, "#c084fc"),
                        ("Total",
                         t_tot, "#fbbf24"),
                    ]):
                    with col:
                        st.markdown(
                            f"""
                            <div class="mb">
                                <div class="mb-v"
                                style="color:{color}">
                                    {val:.0f}ms
                                </div>
                                <div class="mb-l">
                                    {lbl}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — EVALUATION
# ════════════════════════════════════════════
with T2:
    import pandas as pd

    st.markdown("""
    <div class="ib">
        All results below are from
        held-out test sets never seen
        during training.
    </div>
    """, unsafe_allow_html=True)

    et1, et2, et3, et4 = st.tabs([
        "🟢 Classification",
        "🔵 Detection",
        "🟣 Segmentation",
        "🔬 Ablation"
    ])

    with et1:
        st.markdown(
            "##### MobileNetV2 vs Custom CNN")
        for col, (v, l, c) in zip(
                st.columns(3), [
                    ("97.21%",
                     "MobileNetV2 Accuracy",
                     "#00e5cc"),
                    ("81.16%",
                     "Custom CNN Accuracy",
                     "#f87171"),
                    ("0.9905",
                     "Plastic Recall\n(MobileNetV2)",
                     "#5badff"),
                ]):
            with col:
                st.markdown(
                    f"""<div class="mb">
                    <div class="mb-v"
                         style="color:{c}">{v}
                    </div>
                    <div class="mb-l">{l}
                    </div></div>""",
                    unsafe_allow_html=True)

        for p in [
            "results/CustomCNN"
            "_training_curves.png",
            "results/MobileNetV2"
            "_training_curves.png"
        ]:
            if os.path.exists(p):
                nm = ("Custom CNN"
                      if "Custom" in p
                      else "MobileNetV2")
                st.markdown(
                    f"**{nm} Training Curves**")
                st.image(
                    p,
                    use_container_width=True)

        st.dataframe(pd.DataFrame({
            'Model': [
                'CustomCNN','CustomCNN',
                'MobileNetV2','MobileNetV2'],
            'Class': [
                'no-plastic','plastic',
                'no-plastic','plastic'],
            'Train F1': [
                0.969,0.967,0.997,0.996],
            'Val F1': [
                0.986,0.985,0.994,0.994],
            'Test P': [
                0.926,0.742,0.991,0.954],
            'Test R': [
                0.686,0.943,0.955,0.991],
            'Test F1': [
                0.789,0.830,0.972,0.972],
        }), use_container_width=True,
            hide_index=True)

    with et2:
        st.markdown(
            "##### YOLOv8n-seg Test Results")
        for col, (v, l, c) in zip(
                st.columns(4), [
                    ("0.6331","mAP@0.50","#00e5cc"),
                    ("0.5367","mAP@0.50:0.95",
                     "#5badff"),
                    ("0.6556","Precision","#c084fc"),
                    ("0.5827","Recall","#fbbf24"),
                ]):
            with col:
                st.markdown(
                    f"""<div class="mb">
                    <div class="mb-v"
                         style="color:{c}">{v}
                    </div>
                    <div class="mb-l">{l}
                    </div></div>""",
                    unsafe_allow_html=True)

        p = "results/yolo_test_per_class.png"
        if os.path.exists(p):
            st.markdown(
                "**Per-Class — Test Set**")
            st.image(
                p, use_container_width=True)

        st.dataframe(pd.DataFrame({
            'Class': [
                'plastic_bag',
                'plastic_bottle',
                'face_mask','globe',
                'plastic_waste',
                'plastic_cup'],
            'Test Imgs': [72,39,7,2,6,1],
            'AP@0.50': [
                0.9437,0.8007,0.7873,
                0.6392,0.6278,0.0],
            'Precision': [
                0.9462,0.9130,0.8509,
                0.6372,0.5864,0.0],
            'Recall': [
                0.8506,0.6316,0.7170,
                0.6667,0.6304,0.0],
        }), use_container_width=True,
            hide_index=True)

    with et3:
        st.markdown(
            "##### SAM ViT-H Test Results")
        for col, (v, l, c) in zip(
                st.columns(4), [
                    ("0.7922","Mean IoU",
                     "#00e5cc"),
                    ("86.5%","IoU ≥ 0.50",
                     "#5badff"),
                    ("74.0%","IoU ≥ 0.75",
                     "#c084fc"),
                    ("0.2577","Std IoU",
                     "#fbbf24"),
                ]):
            with col:
                st.markdown(
                    f"""<div class="mb">
                    <div class="mb-v"
                         style="color:{c}">{v}
                    </div>
                    <div class="mb-l">{l}
                    </div></div>""",
                    unsafe_allow_html=True)

        p = "results/sam_iou_distribution.png"
        if os.path.exists(p):
            st.markdown(
                "**IoU Distribution — Test Set**")
            st.image(
                p, use_container_width=True)

        st.dataframe(pd.DataFrame({
            'Metric': [
                'Total test images',
                'Processed','Skipped',
                'Mean IoU','Std IoU',
                'Min IoU','Max IoU',
                'IoU ≥ 0.50','IoU ≥ 0.75'],
            'Value': [
                '116','104','12',
                '0.7922','0.2577',
                '0.0000','0.9871',
                '86.5%','74.0%'],
        }), use_container_width=True,
            hide_index=True)

    with et4:
        p = ("results/ablation/"
             "ablation_comparison.png")
        if os.path.exists(p):
            st.image(
                p, use_container_width=True)

        st.markdown("**Ablation 1 — Gate**")
        st.dataframe(pd.DataFrame({
            'Config': [
                'No filter','With filter'],
            'To YOLO': [430, 210],
            'FP Events': [98, 0],
            'Saved': ['—', '51.2%'],
        }), use_container_width=True,
            hide_index=True)

        st.markdown(
            "**Ablation 2 — Segmentation**")
        st.dataframe(pd.DataFrame({
            'Config': [
                'YOLO-only','YOLO+SAM'],
            'Mean IoU': [0.8833, 0.7922],
            'IoU≥0.50': ['96.4%','86.5%'],
            'IoU≥0.75': ['91.1%','74.0%'],
        }), use_container_width=True,
            hide_index=True)

        st.markdown(
            "**End-to-End Analysis**")
        st.dataframe(pd.DataFrame({
            'Stage': [
                '1 Classification',
                '2 Detection',
                '3 Coverage',
                '3 IoU≥0.50'],
            'Value': [
                '0.9905','0.5827',
                '0.8966','0.8650'],
            'Cumulative': [
                '0.9905','0.5772',
                '0.5175','0.4476'],
        }), use_container_width=True,
            hide_index=True)

        st.markdown("""
        <div class="mb" style="margin-top:1rem">
            <div class="mb-v"
                 style="color:#00e5cc">
                ~44.9%
            </div>
            <div class="mb-l">
                End-to-End Success Rate
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════
with T3:
    ca, cb = st.columns([3, 2])

    with ca:
        st.markdown("#### The Pipeline")
        for num, ttl, mdl, prf, sc in [
            ("01","Binary Classification",
             "MobileNetV2 — ImageNet transfer",
             "97.21% test accuracy","sc1"),
            ("02","Object Detection",
             "YOLOv8n-seg — CIoU loss",
             "mAP@0.50 = 0.6331","sc2"),
            ("03","Instance Segmentation",
             "SAM ViT-H — zero-shot prompting",
             "Mean IoU = 0.7922","sc3"),
        ]:
            st.markdown(f"""
            <div class="sc {sc}">
                <div class="sc-num">
                    STAGE {num}
                </div>
                <div class="sc-title">{ttl}
                </div>
                <div class="sc-model">{mdl}
                </div>
                <div style="font-size:.8rem;
                            color:#00e5cc;
                            margin-top:.3rem;
                            font-weight:600">
                    {prf}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with cb:
        st.markdown("#### Key Results")
        for m, v, c in [
            ("Classification","97.21%","#00e5cc"),
            ("Detection mAP","0.6331","#5badff"),
            ("Segmentation IoU","0.7922","#c084fc"),
            ("End-to-End Rate","44.9%","#fbbf24"),
            ("Gate saves","51.2%","#4ade80"),
        ]:
            st.markdown(
                f"""
                <div style="display:flex;
                    justify-content:space-between;
                    align-items:center;
                    padding:.4rem 0;
                    border-bottom:1px solid
                    rgba(255,255,255,0.07)">
                    <span style="color:#94a3b8;
                                 font-size:.8rem">
                        {m}
                    </span>
                    <span style="color:{c};
                        font-family:'Space Mono',
                        monospace;
                        font-size:.82rem;
                        font-weight:700">
                        {v}
                    </span>
                </div>
                """,
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ib">
            <b>Paper:</b> Deep Learning-Based
            Framework for Plastic Debris
            Detection in Dynamic Aquatic
            Ecosystems<br>
            <b>Journal:</b> Turk. J. Math.
            Comput. Sci.<br>
            <b>Author:</b> Hamdi Braiek<br>
            <a href="https://github.com/braiekhamdi/marine-pollution-plastic">
                GitHub Repository
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align:center;
                color:#334155;
                font-size:0.7rem;
                padding:.4rem">
        © 2025 Hamdi Braiek · ESPRIT /
        University of Tunis El Manar
    </div>
    """, unsafe_allow_html=True)