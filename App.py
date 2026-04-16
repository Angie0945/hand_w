import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

# ---------------------------
# CARGAR MODELO UNA VEZ
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# ---------------------------
# PREPROCESAMIENTO
# ---------------------------
def preprocess(img):
    img = ImageOps.grayscale(img)
    img = np.array(img)

    # Invertir colores
    img = cv2.bitwise_not(img)

    # Blur para suavizar
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Threshold
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

# ---------------------------
# SEGMENTACIÓN AVANZADA (CLAVE)
# ---------------------------
def segment_digits_advanced(img):
    # Proyección vertical
    projection = np.sum(img, axis=0)

    segments = []
    start = None

    for i, val in enumerate(projection):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(projection)))

    digits = []

    for (x1, x2) in segments:
        digit = img[:, x1:x2]

        # Filtrar ruido
        if digit.shape[1] > 5:
            digits.append(digit)

    return digits

# ---------------------------
# PREDECIR
# ---------------------------
def predict_digit(digit_img):
    digit_img = cv2.resize(digit_img, (28, 28))
    digit_img = digit_img / 255.0
    digit_img = digit_img.reshape(1, 28, 28, 1)

    pred = model.predict(digit_img, verbose=0)
    return np.argmax(pred)

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Reconocimiento PRO", layout="centered")

st.markdown("## 🔢 Reconocimiento de números pegados")
st.write("Dibuja números juntos (ej: 123, 456) ✍️")

stroke_width = st.slider("✏️ Grosor", 1, 30, 15)

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=400,
    key="canvas",
)

if st.button("🚀 Predecir número"):
    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

        processed = preprocess(img)

        digits = segment_digits_advanced(processed)

        if len(digits) == 0:
            st.warning("No se detectó nada 😅")
        else:
            result = ""

            for d in digits:
                pred = predict_digit(d)
                result += str(pred)

            st.success(f"🔢 Resultado: {result}")

            # Mostrar debug visual
            st.markdown("### 🧩 Segmentación detectada")
            cols = st.columns(len(digits))
            for i, d in enumerate(digits):
                cols[i].image(d, width=80)

    else:
        st.warning("Dibuja algo primero ✍️")

# Sidebar
st.sidebar.title("💡 Tips")
st.sidebar.write("""
✔ Funciona con números pegados  
✔ Mejor si no los haces MUY juntos  
✔ Usa trazos gruesos  
✔ Evita superponerlos completamente  
""")
