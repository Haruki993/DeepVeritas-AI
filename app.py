import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepVeritas AI",
    page_icon="🔍",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e2e8;
    }

    /* ── Rain background on the app root ──────────────────────────────────── */
    .stApp {
        position: relative;
        min-height: 100vh;
        --c: #e879f9;          /* matches the DeepVeritas pink */
        background-color: #000;
        background-image:
            radial-gradient(4px 100px at 0px 235px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 235px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 117.5px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 252px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 252px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 126px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 150px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 150px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 75px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 253px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 253px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 126.5px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 204px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 204px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 102px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 134px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 134px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 67px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 179px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 179px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 89.5px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 299px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 299px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 149.5px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 215px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 215px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 107.5px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 281px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 281px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 140.5px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 158px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 158px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 79px, var(--c) 100%, #0000 150%),
            radial-gradient(4px 100px at 0px 210px, var(--c), #0000),
            radial-gradient(4px 100px at 300px 210px, var(--c), #0000),
            radial-gradient(1.5px 1.5px at 150px 105px, var(--c) 100%, #0000 150%);
        background-size:
            300px 235px, 300px 235px, 300px 235px,
            300px 252px, 300px 252px, 300px 252px,
            300px 150px, 300px 150px, 300px 150px,
            300px 253px, 300px 253px, 300px 253px,
            300px 204px, 300px 204px, 300px 204px,
            300px 134px, 300px 134px, 300px 134px,
            300px 179px, 300px 179px, 300px 179px,
            300px 299px, 300px 299px, 300px 299px,
            300px 215px, 300px 215px, 300px 215px,
            300px 281px, 300px 281px, 300px 281px,
            300px 158px, 300px 158px, 300px 158px,
            300px 210px, 300px 210px, 300px 210px;
        animation: rain 150s linear infinite;
    }

    /* Grid vignette overlay (the "rain through bars" effect) */
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image: radial-gradient(
            ellipse 1.5px 2px at 1.5px 50%,
            #0000 0,
            #0000 90%,
            #000 100%
        );
        background-size: 25px 8px;
    }

    @keyframes rain {
        0% {
            background-position:
                0px 220px,     3px 220px,     151.5px 337.5px,
                25px 24px,     28px 24px,     176.5px 150px,
                50px 16px,     53px 16px,     201.5px 91px,
                75px 224px,    78px 224px,    226.5px 350.5px,
                100px 19px,    103px 19px,    251.5px 121px,
                125px 120px,   128px 120px,   276.5px 187px,
                150px 31px,    153px 31px,    301.5px 120.5px,
                175px 235px,   178px 235px,   326.5px 384.5px,
                200px 121px,   203px 121px,   351.5px 228.5px,
                225px 224px,   228px 224px,   376.5px 364.5px,
                250px 26px,    253px 26px,    401.5px 105px,
                275px 75px,    278px 75px,    426.5px 180px;
        }
        to {
            background-position:
                0px 6800px,    3px 6800px,    151.5px 6917.5px,
                25px 13632px,  28px 13632px,  176.5px 13758px,
                50px 5416px,   53px 5416px,   201.5px 5491px,
                75px 17175px,  78px 17175px,  226.5px 17301.5px,
                100px 5119px,  103px 5119px,  251.5px 5221px,
                125px 8428px,  128px 8428px,  276.5px 8495px,
                150px 9876px,  153px 9876px,  301.5px 9965.5px,
                175px 13391px, 178px 13391px, 326.5px 13540.5px,
                200px 14741px, 203px 14741px, 351.5px 14848.5px,
                225px 18770px, 228px 18770px, 376.5px 18910.5px,
                250px 5082px,  253px 5082px,  401.5px 5161px,
                275px 6375px,  278px 6375px,  426.5px 6480px;
        }
    }

    /* ── All Streamlit content must float above the rain + overlay ────────── */
    .stApp > div {
        position: relative;
        z-index: 1;
    }

    /* ── Rest of DeepVeritas styles ───────────────────────────────────────── */
    .header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        color: #e879f9;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .header p { color: #888; font-size: 0.95rem; margin: 0; }

    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(20, 20, 24, 0.85) !important;
        border: 1.5px dashed #333 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        backdrop-filter: blur(4px);
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #e879f9 !important;
    }

    /* Give main content panels a frosted-glass backing so rain shows through */
    .block-container {
        background: rgba(13, 13, 15, 0.75);
        backdrop-filter: blur(2px);
        border-radius: 18px;
        padding: 2rem !important;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.6rem 2rem 1.2rem 2rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .result-real {
        background: linear-gradient(135deg, #0f2a1f 0%, #0a1a14 100%);
        border: 1.5px solid #22c55e;
    }
    .result-fake {
        background: linear-gradient(135deg, #2a0f0f 0%, #1a0a0a 100%);
        border: 1.5px solid #ef4444;
    }
    .result-uncertain {
        background: linear-gradient(135deg, #2a220f 0%, #1a150a 100%);
        border: 1.5px solid #eab308;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-real .result-label { color: #22c55e; }
    .result-fake .result-label { color: #ef4444; }
    .result-uncertain .result-label { color: #eab308; }
    .result-sub { font-size: 0.88rem; color: #888; margin: 0; }

    .conf-label { font-size: 0.82rem; color: #aaa; margin-bottom: 0.2rem; }
    .conf-pct-real { color: #22c55e; font-weight: 600; }
    .conf-pct-fake { color: #ef4444; font-weight: 600; }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #e879f9;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 2rem 0 0.8rem 0;
    }
    .analysis-caption {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.4rem;
        line-height: 1.5;
    }

    .divider { border: none; border-top: 1px solid #1e1e24; margin: 1.8rem 0; }
    .footer { text-align: center; color: #555; font-size: 0.78rem; padding: 2rem 0 1rem; }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── GitHub button (Uiverse.io by Creatlydev) ────────────────────────── */
    .btn-github {
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border: none;
        text-decoration: none;
        transition: all 0.5s cubic-bezier(0.165, 0.84, 0.44, 1);
        border-radius: 100px;
        font-weight: 800;
        place-content: center;
        padding: 0.75rem 1rem;
        font-size: 0.825rem;
        line-height: 1rem;
        background-color: rgba(0, 0, 0, 0.4);
        box-shadow:
            inset 0 1px 0 0 rgba(255, 255, 255, 0.04),
            inset 0 0 0 1px rgba(255, 255, 255, 0.04);
        color: #fff;
        width: fit-content;
        margin: 0 auto;
    }
    .btn-github:hover {
        box-shadow:
            inset 0 1px 0 0 rgba(255, 255, 255, 0.08),
            inset 0 0 0 1px rgba(252, 232, 3, 0.08);
        color: #fce803;
        transform: translate(0, -0.25rem);
        background-color: rgba(0, 0, 0, 0.5);
    }
    .btn-github svg { width: 1.1rem; height: 1.1rem; fill: currentColor; }
    .github-wrap { display: flex; justify-content: center; padding: 1rem 0 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#141418",
    "axes.facecolor":    "#141418",
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   "#aaa",
    "xtick.color":       "#666",
    "ytick.color":       "#666",
    "text.color":        "#e2e2e8",
    "grid.color":        "#222",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL image for st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH = "./models"
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH, attn_implementation="eager")
    model.eval()
    return model, processor

# ── Inference + attention ─────────────────────────────────────────────────────
def predict(image: Image.Image, model, processor):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    probs     = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx  = torch.argmax(probs).item()
    label     = model.config.id2label[pred_idx]
    is_real   = label.lower() in ("realism", "real")
    conf_real = probs[0].item() if model.config.id2label[0].lower() in ("realism", "real") else probs[1].item()
    conf_fake = 1.0 - conf_real

    # ── Attention map with fallback ───────────────────────────────────────────
    attn_map = None

    if outputs.attentions and len(outputs.attentions) > 0:
        last_attn = outputs.attentions[-1]
        cls_attn  = last_attn[0].mean(0)[0, 1:]
        grid_size = int(cls_attn.shape[0] ** 0.5)
        attn_map  = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    else:
        img_tensor = inputs["pixel_values"].requires_grad_(True)
        logits     = model(pixel_values=img_tensor).logits
        logits[0, pred_idx].backward()
        saliency   = img_tensor.grad[0].abs().mean(0)
        attn_map   = saliency.cpu().numpy()

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return is_real, conf_real, conf_fake, attn_map

# ── Analysis plots ────────────────────────────────────────────────────────────
def plot_attention(image: Image.Image, attn_map: np.ndarray) -> Image.Image:
    img_rgb  = np.array(image.convert("RGB").resize((224, 224)))
    heat     = Image.fromarray(attn_map).resize((224, 224), Image.BILINEAR)
    heat_arr = np.array(heat)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img_rgb);                axes[0].set_title("Original",  fontsize=9); axes[0].axis("off")
    axes[1].imshow(heat_arr, cmap="magma"); axes[1].set_title("Attention", fontsize=9); axes[1].axis("off")

    colored = cm.magma(heat_arr)[:, :, :3]
    overlay = np.clip(0.55 * img_rgb / 255.0 + 0.45 * colored, 0, 1)
    axes[2].imshow(overlay);               axes[2].set_title("Overlay",   fontsize=9); axes[2].axis("off")

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result


def plot_fft(image: Image.Image) -> Image.Image:
    gray     = np.array(image.convert("L").resize((224, 224))).astype(np.float32)
    fft      = np.fft.fftshift(np.fft.fft2(gray))
    spectrum = np.log1p(np.abs(fft))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    im0 = axes[0].imshow(spectrum, cmap="inferno")
    axes[0].set_title("FFT Magnitude Spectrum", fontsize=9); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    cy, cx  = np.array(spectrum.shape) // 2
    Y, X    = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
    R       = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    radial  = np.bincount(R.ravel(), weights=spectrum.ravel()) / (np.bincount(R.ravel()) + 1e-8)
    axes[1].plot(np.arange(len(radial))[:80], radial[:80], color="#e879f9", linewidth=1.5)
    axes[1].set_title("Radial Power Profile", fontsize=9)
    axes[1].set_xlabel("Spatial frequency"); axes[1].set_ylabel("Mean log power")
    axes[1].grid(True)

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result


def plot_histogram(image: Image.Image) -> Image.Image:
    img_arr = np.array(image.convert("RGB").resize((224, 224)))
    colors  = [("#ef4444", "Red"), ("#22c55e", "Green"), ("#60a5fa", "Blue")]

    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (color, label) in enumerate(colors):
        hist, bins = np.histogram(img_arr[:, :, i].ravel(), bins=64, range=(0, 255))
        centers    = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, hist, color=color, label=label, linewidth=1.5, alpha=0.85)

    ax.set_title("RGB Pixel Distribution", fontsize=9)
    ax.set_xlabel("Pixel value (0–255)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8, framealpha=0.2); ax.grid(True)

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result


def plot_noise(image: Image.Image) -> Image.Image:
    from scipy.ndimage import gaussian_filter
    gray   = np.array(image.convert("L").resize((224, 224))).astype(np.float32)
    smooth = gaussian_filter(gray, sigma=2)
    noise  = gray - smooth

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    im0 = axes[0].imshow(noise, cmap="RdBu_r", vmin=-30, vmax=30)
    axes[0].set_title("Noise Residual", fontsize=9); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(noise.ravel(), bins=80, color="#e879f9", alpha=0.8, edgecolor="none")
    axes[1].set_title("Noise Distribution", fontsize=9)
    axes[1].set_xlabel("Residual value"); axes[1].set_ylabel("Count")
    axes[1].grid(True)

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🔍 DeepVeritas AI</h1>
    <p>Upload an image to detect if it's real or AI-generated</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model, processor = load_model()

uploaded = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width="stretch")

    with st.spinner("Analyzing..."):
        is_real, conf_real, conf_fake, attn_map = predict(image, model, processor)

    max_conf = max(conf_real, conf_fake)
    if max_conf < 0.70:
        verdict    = "UNCERTAIN"
        card_class = "result-uncertain"
        icon       = "🤔"
        subtext    = "The model is uncertain. This could be a heavily edited real photo or a highly realistic deepfake."
    else:
        verdict    = "REAL"        if is_real else "FAKE"
        card_class = "result-real" if is_real else "result-fake"
        icon       = "✅"           if is_real else "⚠️"
        subtext    = "This image appears to be a genuine photograph." if is_real \
                     else "This image appears to be AI-generated or manipulated."

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-label">{icon} {verdict}</div>
        <div class="result-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_r, col_f = st.columns(2)
    with col_r:
        st.markdown(f'<div class="conf-label">Real &nbsp;<span class="conf-pct-real">{conf_real*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(float(conf_real))
    with col_f:
        st.markdown(f'<div class="conf-label">Fake &nbsp;<span class="conf-pct-fake">{conf_fake*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(float(conf_fake))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧠 Deep Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Generating analysis plots..."):
        attn_img  = plot_attention(image, attn_map)
        fft_img   = plot_fft(image)
        hist_img  = plot_histogram(image)
        noise_img = plot_noise(image)

    st.image(attn_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>Attention Heatmap</b> — Shows which regions of the image the model focused on most.
    Bright areas = high attention. AI-generated images often show uniform or unnatural attention patterns
    (e.g. focused on textures or backgrounds instead of meaningful features).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(fft_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>FFT Frequency Spectrum</b> — Converts the image into frequency space.
    AI-generated images often show grid-like artefacts or unnatural spikes at regular intervals
    caused by upsampling in the generator network. The radial power profile shows how energy
    is distributed across spatial frequencies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(hist_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>RGB Pixel Distribution</b> — Shows how pixel values are spread across each colour channel.
    Real photographs tend to have irregular, camera-noise-influenced distributions.
    AI-generated images often produce unnaturally smooth or symmetrical histograms.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(noise_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>Noise Residual</b> — Extracts high-frequency noise by subtracting a smoothed version of the image.
    Real cameras produce random grain. AI generators often leave behind structured, repeating, or
    suspiciously regular noise patterns that are invisible to the eye but visible here.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    with st.expander("Image details"):
        st.write(f"**Filename:** {uploaded.name}")
        st.write(f"**Dimensions:** {image.width} × {image.height} px")
        st.write(f"**Format:** {image.format or uploaded.type}")
        st.write(f"**Real confidence:** {conf_real*100:.2f}%")
        st.write(f"**Fake confidence:** {conf_fake*100:.2f}%")

st.markdown('<div class="footer">DeepVeritas AI v2 &middot; High-Res ViT &middot; Multi-Generator Dataset</div>', unsafe_allow_html=True)

# ── GitHub link button ──────────────────────────────────────────────────────
st.markdown("""
<div class="github-wrap">
    <a class="btn-github" href="https://github.com/Haruki993/DeepVeritas-AI" target="_blank">
        <svg viewBox="0 0 16 16" version="1.1" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
            0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
            -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07
            -1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
            a7.6 7.6 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
            .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
            0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
        <span>Haruki993/DeepVeritas-AI</span>
    </a>
</div>
""", unsafe_allow_html=True)
