import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. THE ADVANCED ENGINE ---
class DIPEngine:
    @staticmethod
    def apply_sepia(img):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_beauty_smooth(img, power):
        return cv2.bilateralFilter(img, 9, power, power)

    @staticmethod
    def apply_sketch(img, detail):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (detail, detail), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def detect_faces(img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        img_copy = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)
        return img_copy, len(faces)

    # --- NEW CONCEPTS: INTERPOLATION & LAPLACIAN ---
    @staticmethod
    def apply_interpolation(img, scale, method_name):
        """
        Geometric Transformation: Estimates new pixel values when resizing.
        """
        methods = {
            "Nearest (Fast/Pixelated)": cv2.INTER_NEAREST,
            "Bilinear (Balanced)": cv2.INTER_LINEAR,
            "Bicubic (High Quality)": cv2.INTER_CUBIC
        }
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height), interpolation=methods[method_name])

    @staticmethod
    def apply_laplacian(img):
        """
        Frequency Analysis: High-pass filter using 2nd order derivatives.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CV_64F captures negative slopes, which we then take absolute value of
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.absolute(lap))
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def apply_canny(img, t1, t2):
        edges = cv2.Canny(img, t1, t2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# --- 2. UI & STYLE ---
st.set_page_config(page_title="Ultimate DIP Studio", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #0095f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INPUT HANDLING ---
st.title("🚀 Ultimate Digital Image Studio")
st.sidebar.markdown("### 🛠️ Input Source")
input_method = st.sidebar.radio("Method", ("Upload File", "Live Camera"))

final_img = None
if input_method == "Upload File":
    uploaded = st.sidebar.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        final_img = np.array(Image.open(uploaded))
else:
    captured = st.camera_input("Take a photo")
    if captured:
        final_img = np.array(Image.open(captured))

# --- 4. FILTER EXECUTION ---
if final_img is not None:
    st.sidebar.markdown("---")
    filter_type = st.sidebar.selectbox("Select Transformation", 
        ["Original", "Interpolation Rescale", "Laplacian Edges", "Face Detection", "Pencil Sketch", "Beauty Smooth", "Vintage Sepia", "Canny Edges"])
    
    engine = DIPEngine()
    
    if filter_type == "Interpolation Rescale":
        sc = st.sidebar.slider("Scale Factor", 0.2, 3.0, 1.0)
        met = st.sidebar.radio("Interpolation Algorithm", ("Nearest (Fast/Pixelated)", "Bilinear (Balanced)", "Bicubic (High Quality)"))
        result = engine.apply_interpolation(final_img, sc, met)
        st.sidebar.info(f"Resolution: {result.shape[1]} x {result.shape[0]}")
    
    elif filter_type == "Laplacian Edges":
        result = engine.apply_laplacian(final_img)
        st.sidebar.write("Concept: 2nd Order Derivative")
        
    elif filter_type == "Face Detection":
        result, count = engine.detect_faces(final_img)
        st.sidebar.success(f"Detected: {count}")
        
    elif filter_type == "Pencil Sketch":
        det = st.sidebar.slider("Detail", 11, 51, 21, step=2)
        result = engine.apply_sketch(final_img, det)
        
    elif filter_type == "Beauty Smooth":
        pwr = st.sidebar.slider("Power", 10, 150, 75)
        result = engine.apply_beauty_smooth(final_img, pwr)
        
    elif filter_type == "Vintage Sepia":
        result = engine.apply_sepia(final_img)
        
    elif filter_type == "Canny Edges":
        t1 = st.sidebar.slider("T1", 0, 255, 100)
        t2 = st.sidebar.slider("T2", 0, 255, 200)
        result = engine.apply_canny(final_img, t1, t2)
    else:
        result = final_img

    # Display Side-by-Side
    c1, c2 = st.columns(2)
    with c1: st.image(final_img, caption="Source", use_container_width=True)
    with c2: st.image(result, caption=filter_type, use_container_width=True)

    # Histogram Analysis
    st.subheader("📊 Frequency Distribution")
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, col in enumerate(['red', 'green', 'blue']):
        hist = cv2.calcHist([result], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_facecolor('#0e1117'); fig.patch.set_facecolor('#0e1117'); ax.tick_params(colors='white')
    st.pyplot(fig)
else:
    st.info("Awaiting Input...")
