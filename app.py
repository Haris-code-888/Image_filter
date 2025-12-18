import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. THE ENGINE (Advanced Logic) ---
class AdvancedFilterEngine:
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

# --- 2. THE UI STYLING ---
st.set_page_config(page_title="DIP Studio Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #0095f6; color: white; border: none; }
    h1, h2, h3 { color: #0095f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE INTERFACE ---
st.title("🎨 Advanced Digital Image Studio")
st.sidebar.markdown("### 🛠️ Input Source")

# Select between Upload or Camera
input_method = st.sidebar.radio("Select Input Method", ("Upload File", "Live Camera Capture"))

final_img = None

if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        raw_img = Image.open(uploaded_file)
        final_img = np.array(raw_img)
else:
    captured_photo = st.camera_input("Take a snapshot to process")
    if captured_photo:
        raw_img = Image.open(captured_photo)
        final_img = np.array(raw_img)

# --- 4. PROCESSING LOGIC ---
if final_img is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Filter Gallery")
    filter_type = st.sidebar.selectbox("Choose Filter", ["Original", "Vintage Sepia", "Beauty Smooth", "Pencil Sketch", "Face Detection"])
    
    strength = 75
    if filter_type == "Beauty Smooth":
        strength = st.sidebar.slider("Smoothing Power", 10, 150, 75)
    elif filter_type == "Pencil Sketch":
        strength = st.sidebar.select_slider("Sketch Detail", options=[15, 21, 31, 51])

    engine = AdvancedFilterEngine()
    with st.spinner('Applying advanced math...'):
        if filter_type == "Original":
            result = final_img
        elif filter_type == "Vintage Sepia":
            result = engine.apply_sepia(final_img)
        elif filter_type == "Beauty Smooth":
            result = engine.apply_beauty_smooth(final_img, strength)
        elif filter_type == "Pencil Sketch":
            result = engine.apply_sketch(final_img, strength)
        elif filter_type == "Face Detection":
            result, count = engine.detect_faces(final_img)
            st.sidebar.info(f"Faces Detected: {count}")

    # Layout: Side-by-Side Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Image")
        st.image(final_img, use_container_width=True)
    with col2:
        st.subheader(f"Processed: {filter_type}")
        st.image(result, use_container_width=True)

    # Advanced Analysis: Histogram
    st.markdown("---")
    st.subheader("📊 Color Distribution Analysis (Processed)")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ('red', 'green', 'blue')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([result], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, alpha=0.7)
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    st.pyplot(fig)
    
    # Download Button
    result_pil = Image.fromarray(result)
    result_pil.save("final_output.png")
    with open("final_output.png", "rb") as f:
        st.download_button("💾 Download Processed Image", f, file_name="processed_image.png")
else:
    st.info("Please upload an image or take a photo to begin.")
