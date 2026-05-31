"""
Kidney Stone AI Diagnostic System
Full pipeline: Classification → Detection → Segmentation
"""
import streamlit as st
import torch
import torchvision
import cv2
import numpy as np
import os, sys
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.append(os.path.join(os.path.dirname(__file__), 'classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))
from train import get_model as get_classifier

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kidney Stone AI System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; }
    
    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #0d1b2a 50%, #162032 100%);
        border: 1px solid #2a3a5c;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0; padding: 0;
    }
    .hero-header p {
        color: #8a9bb5;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2d45, #162032);
        border: 1px solid #2a3a5c;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .label { color: #8a9bb5; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #4fc3f7; font-size: 1.9rem; font-weight: 700; }
    
    /* Result boxes */
    .result-stone {
        background: linear-gradient(135deg, #3d1515, #5c2020);
        border: 2px solid #e53935;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ff6b6b;
    }
    .result-normal {
        background: linear-gradient(135deg, #0d2e1a, #1b5e20);
        border: 2px solid #43a047;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #69f0ae;
    }
    .section-title {
        color: #4fc3f7;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #2a3a5c;
        padding-bottom: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e2d45;
        border-radius: 8px;
        color: #8a9bb5;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1565c0, #7c4dff) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES  = ["Non-Stone", "Stone"]

# ─── Model loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    model = get_classifier(len(CLASSES))
    path = "models/stone_classifier.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_detector():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    path = "models/faster_rcnn_stone.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_unet():
    from unet_model import UNet
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    path = "models/unet_stone.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

# ─── Inference helpers ────────────────────────────────────────────────────────
def classify(image: Image.Image, model):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    t = tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(t), dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASSES[pred.item()], conf.item(), probs[0].cpu().tolist()

def detect(image: Image.Image, model, threshold=0.5):
    tf = transforms.ToTensor()
    t  = tf(image).to(DEVICE)
    with torch.no_grad():
        preds = model([t])[0]
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    stone_count = 0
    boxes_kept  = []
    for box, score in zip(preds["boxes"], preds["scores"]):
        if score.item() >= threshold:
            x1,y1,x2,y2 = map(int, box.tolist())
            cv2.rectangle(img_cv, (x1,y1),(x2,y2),(0,255,127), 3)
            cv2.putText(img_cv, f"{score.item()*100:.1f}%", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,127), 2)
            stone_count += 1
            boxes_kept.append(score.item())
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), stone_count, boxes_kept

def segment(image: Image.Image, model):
    IMG_SIZE = 256
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    orig_w, orig_h = image.size
    img_resized    = image.resize((IMG_SIZE, IMG_SIZE))
    t              = tf(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(t)
    mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    img_cv     = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    green      = np.zeros_like(img_cv)
    green[mask_full == 1] = (0, 255, 100)
    overlay    = cv2.addWeighted(img_cv, 0.65, green, 0.35, 0)
    coverage   = (mask_full.sum() / mask_full.size) * 100

    return (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            mask_full * 255,
            coverage)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🫁 Kidney Stone AI Diagnostic System</h1>
  <p>End-to-end computer vision pipeline · Classification · Object Detection · Segmentation</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    det_threshold = st.slider("Detection Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.markdown("### 📋 Model Status")

    cls_ok  = os.path.exists("models/stone_classifier.pth")
    det_ok  = os.path.exists("models/faster_rcnn_stone.pth")
    seg_ok  = os.path.exists("models/unet_stone.pth")

    st.markdown(f"{'✅' if cls_ok else '❌'} Classifier")
    st.markdown(f"{'✅' if det_ok else '❌'} Detector (Faster R-CNN)")
    st.markdown(f"{'✅' if seg_ok else '❌'} Segmenter (U-Net)")
    st.divider()
    st.caption("Upload an image to analyse it using all three models.")

# ─── File Upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a kidney CT / ultrasound image",
                             type=["jpg","jpeg","png","bmp","tif","tiff"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    tab1, tab2, tab3 = st.tabs([
        "🔬 Classification", "📦 Object Detection", "🎨 Segmentation"
    ])

    # ══ Tab 1: Classification ═════════════════════════════════════════════════
    with tab1:
        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_res:
            st.markdown('<div class="section-title">Classification Result</div>', unsafe_allow_html=True)
            try:
                cls_model = load_classifier()
                label, conf, probs = classify(image, cls_model)

                if label == "Stone":
                    st.markdown(f'<div class="result-stone">🔴 Kidney Stone Detected</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-normal">🟢 No Stone Detected</div>',
                                unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="metric-card"><div class="label">Confidence</div><div class="value">{conf*100:.1f}%</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="label">Prediction</div><div class="value">{label}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>**Class Probabilities**", unsafe_allow_html=True)
                for cls_name, prob in zip(CLASSES, probs):
                    st.progress(prob, text=f"{cls_name}: {prob*100:.1f}%")

            except Exception as e:
                st.error(f"Classification error: {e}")
                st.info("Train the classifier first: `python classification/train.py`")

    # ══ Tab 2: Detection ══════════════════════════════════════════════════════
    with tab2:
        col_img2, col_res2 = st.columns([1, 1], gap="large")
        with col_img2:
            st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_res2:
            st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
            try:
                det_model = load_detector()
                det_img, count, scores = detect(image, det_model, det_threshold)
                st.image(det_img, caption="Detected Bounding Boxes (green)", use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="metric-card"><div class="label">Stones Found</div><div class="value">{count}</div></div>', unsafe_allow_html=True)
                avg_conf = np.mean(scores)*100 if scores else 0
                c2.markdown(f'<div class="metric-card"><div class="label">Avg Confidence</div><div class="value">{avg_conf:.1f}%</div></div>', unsafe_allow_html=True)

                if count > 0:
                    st.markdown(f'<div class="result-stone">⚠️ {count} Stone Region(s) Detected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-normal">✅ No Stones Detected above threshold</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Detection error: {e}")
                st.info("Train the detector first and place `faster_rcnn_stone.pth` in `models/`")

    # ══ Tab 3: Segmentation ═══════════════════════════════════════════════════
    with tab3:
        col_img3, col_res3 = st.columns([1, 1], gap="large")
        with col_img3:
            st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_res3:
            st.markdown('<div class="section-title">Segmentation Result</div>', unsafe_allow_html=True)
            try:
                seg_model = load_unet()
                overlay, mask, coverage = segment(image, seg_model)
                st.image(overlay, caption="Green = Segmented Stone Region", use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="metric-card"><div class="label">Stone Coverage</div><div class="value">{coverage:.1f}%</div></div>', unsafe_allow_html=True)
                mask_pil = Image.fromarray(mask)
                import io
                buf = io.BytesIO()
                mask_pil.save(buf, format="PNG")
                c2.download_button("⬇️ Download Mask", buf.getvalue(),
                                   file_name="stone_mask.png", mime="image/png")

                if coverage > 1.0:
                    st.markdown(f'<div class="result-stone">🔴 Stone Region Segmented ({coverage:.1f}% of image)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-normal">🟢 Minimal / No Stone Region Found</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Segmentation error: {e}")
                st.info("Train the U-Net first and place `unet_stone.pth` in `models/`")

else:
    # Placeholder state
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #8a9bb5;'>
        <div style='font-size:4rem'>🫁</div>
        <h3 style='color:#4fc3f7;'>Upload an Image to Begin Analysis</h3>
        <p>Supports: JPG, PNG, BMP, TIFF · CT Scans · Ultrasound</p>
    </div>
    """, unsafe_allow_html=True)
