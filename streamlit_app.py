"""
streamlit_app.py — Interactive Off-Road Segmentation Demo Tool

Usage:
    streamlit run streamlit_app.py

Features:
    - Upload any image (JPG, PNG, WEBP)
    - Select from available checkpoints
    - See original image + color-coded segmentation mask side-by-side
    - Per-class IoU/legend breakdown
    - TTA toggle for best quality
    - Download the prediction mask
"""

import os
import ssl
import sys
import glob
import io
import time

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Offroad Segmentation Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

COLOR_PALETTE = np.array([
    [0,   0,   0],      # Background — black
    [34,  139, 34],     # Trees — forest green
    [0,   255, 0],      # Lush Bushes — lime
    [210, 180, 140],    # Dry Grass — tan
    [139, 90,  43],     # Dry Bushes — brown
    [128, 128, 0],      # Ground Clutter — olive
    [255, 20,  147],    # Flowers — deep pink
    [139, 69,  19],     # Logs — saddle brown
    [128, 128, 128],    # Rocks — gray
    [160, 82,  45],     # Landscape — sienna
    [135, 206, 235],    # Sky — sky blue
], dtype=np.uint8)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES = 11

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #e94560;
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.15);
    }
    .main-header h1 { color: #ffffff; font-size: 2.4rem; margin: 0; font-weight: 700; }
    .main-header p  { color: #adb5bd; margin: 0.5rem 0 0; font-size: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border: 1px solid #363d5e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .label { color: #8892b0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #ccd6f6; font-size: 1.8rem; font-weight: 700; }

    .class-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #1e2130;
        border: 1px solid #363d5e;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.82rem;
        color: #ccd6f6;
    }
    .color-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
    }

    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c33355);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(233,69,96,0.4); }

    .success-box {
        background: linear-gradient(135deg, #0d2b1a, #1a3a2a);
        border: 1px solid #2ea868;
        border-radius: 10px;
        padding: 1rem;
        color: #81c995;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #2b1a0d, #3a2a1a);
        border: 1px solid #e8924b;
        border-radius: 10px;
        padding: 1rem;
        color: #f0b57a;
        margin: 0.5rem 0;
    }

    div[data-testid="stSidebar"] { background: #12151f; }
    div[data-testid="stSidebar"] .block-container { padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Load model ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str):
    """Load model from checkpoint. Cached so it's only done once per checkpoint."""
    try:
        # Add project dir to path so we can import project modules
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from models import SegmentationModel

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu"
        )

        model = SegmentationModel(use_aux=False)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        epoch   = checkpoint.get("epoch", "?")
        best_iou = checkpoint.get("best_iou", None)
        return model, device, epoch, best_iou

    except Exception as e:
        return None, None, None, str(e)


# ─── Helper: Preprocess image ─────────────────────────────────────────────────
def preprocess(pil_img: Image.Image, size=(512, 512)) -> torch.Tensor:
    img = pil_img.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


# ─── Helper: Run inference (with optional TTA) ────────────────────────────────
@torch.no_grad()
def run_inference(model, tensor, device, use_tta=False):
    tensor = tensor.to(device)

    if use_tta:
        prob1 = F.softmax(model(tensor), dim=1)
        prob2 = F.softmax(torch.flip(model(torch.flip(tensor, dims=[3])), dims=[3]), dim=1)
        prob3 = F.softmax(torch.flip(model(torch.flip(tensor, dims=[2])), dims=[2]), dim=1)
        probs = (prob1 + prob2 + prob3) / 3.0
    else:
        probs = F.softmax(model(tensor), dim=1)

    pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    confs = probs.squeeze(0).cpu().numpy()
    return pred, confs


# ─── Helper: Colorize mask ────────────────────────────────────────────────────
def colorize_mask(pred_mask: np.ndarray, alpha_orig=None, orig_img=None, alpha=0.5):
    """Convert integer mask → RGB color image."""
    h, w = pred_mask.shape
    color_mask = COLOR_PALETTE[pred_mask.clip(0, NUM_CLASSES - 1)]

    if orig_img is not None and alpha_orig is not None:
        orig_arr = np.array(orig_img.resize((w, h))).astype(np.float32)
        overlay = orig_arr * (1 - alpha) + color_mask.astype(np.float32) * alpha
        return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    return Image.fromarray(color_mask)


# ─── Helper: Per-class pixel stats ────────────────────────────────────────────
def class_pixel_stats(pred_mask: np.ndarray):
    total = pred_mask.size
    stats = {}
    for i, name in enumerate(CLASS_NAMES):
        if i == 0:  # skip background
            continue
        count = int((pred_mask == i).sum())
        stats[name] = {"count": count, "pct": count / total * 100, "color": COLOR_PALETTE[i]}
    return dict(sorted(stats.items(), key=lambda x: -x[1]["pct"]))


# ─── Get available checkpoints ────────────────────────────────────────────────
def get_checkpoints():
    patterns = [
        os.path.join(CHECKPOINT_DIR, "*.pth"),
        os.path.join(CHECKPOINT_DIR, "**", "*.pth"),
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p, recursive=True))
    return sorted(set(found))


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    # ── Checkpoint selection ──
    st.markdown("**📦 Model Checkpoint**")
    checkpoints = get_checkpoints()

    if not checkpoints:
        st.error(f"No `.pth` files found in:\n`{CHECKPOINT_DIR}`")
        st.stop()

    # Allow custom path too
    custom_path = st.text_input("Or paste custom path:", placeholder="/path/to/model.pth")

    ckpt_display = {os.path.basename(c): c for c in checkpoints}
    if custom_path and os.path.exists(custom_path):
        ckpt_display[f"📎 {os.path.basename(custom_path)}"] = custom_path

    selected_label = st.selectbox(
        "Select checkpoint",
        options=list(ckpt_display.keys()),
        help="Choose which saved model to run inference with"
    )
    selected_ckpt = ckpt_display[selected_label]

    st.caption(f"`{selected_ckpt}`")
    ckpt_size_mb = os.path.getsize(selected_ckpt) / (1024**2)
    st.caption(f"Size: **{ckpt_size_mb:.1f} MB**")

    st.divider()

    # ── Inference settings ──
    st.markdown("**🔬 Inference Settings**")
    use_tta = st.toggle("Test-Time Augmentation (TTA)", value=False,
                        help="Averages predictions from 3 views. +2-4% IoU but 3x slower.")
    show_overlay = st.toggle("Show as overlay on image", value=True,
                             help="Blend mask on top of original image")
    overlay_alpha = 0.45
    if show_overlay:
        overlay_alpha = st.slider("Overlay opacity", 0.2, 0.8, 0.45, 0.05)

    img_size_choice = st.selectbox("Input resolution", ["512×512", "448×448", "384×384"],
                                   index=0, help="Larger = better quality, slower")
    size_map = {"512×512": (512, 512), "448×448": (448, 448), "384×384": (384, 384)}
    img_size = size_map[img_size_choice]

    st.divider()
    st.markdown("**ℹ️ Classes (10 active)**")
    for i in range(1, NUM_CLASSES):
        rgb = COLOR_PALETTE[i]
        hex_col = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        st.markdown(
            f'<div class="class-pill">'
            f'<span class="color-dot" style="background:{hex_col}"></span>'
            f'{CLASS_NAMES[i]}</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>Off-Road Segmentation Demo</h1>
    <p>Upload any desert scene and see per-pixel terrain classification in real time</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ──
with st.spinner(f"Loading **{selected_label}**…"):
    model, device, epoch, best_iou = load_model(selected_ckpt)

if model is None:
    st.error(f"❌ Failed to load model:\n```\n{best_iou}\n```")
    st.stop()

# Show model info bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f'<div class="metric-card"><div class="label">Checkpoint</div><div class="value" style="font-size:1rem">{selected_label}</div></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown(f'<div class="metric-card"><div class="label">Trained Epoch</div><div class="value">{epoch}</div></div>', unsafe_allow_html=True)
with col_m3:
    iou_str = f"{best_iou:.4f}" if isinstance(best_iou, float) else "—"
    st.markdown(f'<div class="metric-card"><div class="label">Best mIoU</div><div class="value">{iou_str}</div></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown(f'<div class="metric-card"><div class="label">Device</div><div class="value">{str(device).upper()}</div></div>', unsafe_allow_html=True)

st.divider()

# ── File uploader ──
uploaded = st.file_uploader(
    "📸 Upload an image (JPG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload any desert scene image to run segmentation on it"
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = pil_img.size

    st.markdown(f"**Uploaded:** `{uploaded.name}` · {orig_w}×{orig_h}px · {uploaded.size/1024:.1f} KB")

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("🚀 Run Segmentation", use_container_width=True)

    # Initialize session state for this image if needed
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded.name:
        st.session_state.results = None
        st.session_state.last_uploaded = uploaded.name

    if run_btn:
        with st.spinner(f"Running inference {'(TTA enabled)' if use_tta else ''}…"):
            t0 = time.time()
            tensor = preprocess(pil_img, size=img_size)
            pred_mask, confs = run_inference(model, tensor, device, use_tta=use_tta)
            elapsed = time.time() - t0

        # Resize mask back to original image size
        pred_pil_small = Image.fromarray(pred_mask.astype(np.uint8))
        pred_resized = np.array(pred_pil_small.resize((orig_w, orig_h), Image.NEAREST))

        # Generate outputs
        color_mask = colorize_mask(pred_resized)
        
        # Save to session_state so it survives the download button click
        st.session_state.results = {
            "pred_resized": pred_resized,
            "color_mask": color_mask,
            "confs": confs,
            "elapsed": elapsed,
            "use_tta": use_tta,
            "img_size": img_size
        }

    # If we have results for the current image, display them
    if st.session_state.results is not None:
        res = st.session_state.results
        pred_resized = res["pred_resized"]
        color_mask = res["color_mask"]
        confs = res["confs"]
        
        if show_overlay:
            display_mask = colorize_mask(pred_resized, orig_img=pil_img, alpha_orig=True, alpha=overlay_alpha)
        else:
            display_mask = color_mask

        st.markdown(f'<div class="success-box"> Inference complete in <strong>{res["elapsed"]:.2f}s</strong> · Resolution: {res["img_size"][0]}×{res["img_size"][1]} · TTA: {"ON" if res["use_tta"] else "OFF"}</div>', unsafe_allow_html=True)
        
        # ── Side-by-side comparison ──
        st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>📷 Before & After Processing</h2>", unsafe_allow_html=True)
        img_col, color_col, overlay_col = st.columns(3)
        
        with img_col:
            st.markdown("<h4 style='text-align: center;'>Original Image</h4>", unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            
        with color_col:
            st.markdown("<h4 style='text-align: center;'>Color Mask</h4>", unsafe_allow_html=True)
            st.image(color_mask, use_container_width=True)
            
        with overlay_col:
            st.markdown("<h4 style='text-align: center;'>Overlay Blend</h4>", unsafe_allow_html=True)
            st.image(display_mask, use_container_width=True)

        st.divider()

        # ── Per-class breakdown ──
        st.markdown("### 📊 Detected Class Coverage")
        stats = class_pixel_stats(pred_resized)

        active_classes = {k: v for k, v in stats.items() if v["pct"] > 0.05}

        if active_classes:
            cols = st.columns(min(len(active_classes), 5))
            for idx, (name, info) in enumerate(list(active_classes.items())[:10]):
                rgb = info["color"]
                hex_col = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                with cols[idx % 5]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div style="width:24px;height:24px;border-radius:6px;background:{hex_col};margin:0 auto 0.4rem"></div>'
                        f'<div class="label">{name}</div>'
                        f'<div class="value">{info["pct"]:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Bar chart
            st.markdown("#### Pixel Distribution")
            bar_data = {k: v["pct"] for k, v in active_classes.items()}
            st.bar_chart(bar_data)

        st.divider()

        # ── Confidence heatmap for top class ──
        st.markdown("### 🌡️ Model Confidence (Max Class Probability per Pixel)")
        max_conf = confs.max(axis=0)
        # Resize conf map
        conf_pil = Image.fromarray((max_conf * 255).astype(np.uint8))
        conf_resized = conf_pil.resize((orig_w, orig_h), Image.BILINEAR)
        st.image(conf_resized, use_container_width=True,
                 caption="White = very confident | Dark = uncertain boundary regions")

        st.divider()

        # ── Downloads ──
        st.markdown("### 💾 Downloads")
        dl1, dl2, dl3 = st.columns(3)

        def pil_to_bytes(img, fmt="PNG"):
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            return buf.getvalue()

        with dl1:
            st.download_button(
                "⬇️ Download Color Mask",
                data=pil_to_bytes(color_mask),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_mask.png",
                mime="image/png",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                "⬇️ Download Overlay",
                data=pil_to_bytes(display_mask),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_overlay.png",
                mime="image/png",
                use_container_width=True
            )
        with dl3:
            raw_mask = Image.fromarray(pred_resized.astype(np.uint8))
            st.download_button(
                "⬇️ Download Raw Mask",
                data=pil_to_bytes(raw_mask),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_raw.png",
                mime="image/png",
                use_container_width=True
            )

else:
    # Placeholder instructions
    st.markdown("""
    <div style="
        border: 2px dashed #363d5e;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: #12151f;
        margin: 2rem 0;
    ">
        <div style="font-size: 3rem">📸</div>
        <h3 style="color: #ccd6f6; margin: 1rem 0 0.5rem">Upload an image to get started</h3>
        <p style="color: #8892b0">
            Upload any desert scene image (JPG/PNG/WEBP) using the file uploader above.<br>
            Select a model checkpoint from the sidebar, then click <strong>Run Segmentation</strong>.
        </p>
        <p style="color: #636e8a; font-size: 0.85rem; margin-top: 1.5rem">
            💡 Enable <strong>TTA</strong> in the sidebar for +2-4% better accuracy<br>
            💡 Toggle <strong>Overlay</strong> to blend the mask on top of your original image
        </p>
    </div>
    """, unsafe_allow_html=True)
