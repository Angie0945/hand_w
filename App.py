import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

# Cargar modelo UNA SOLA VEZ (más rápido)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# ---------------------------
# PREPROCESAMIENTO DE IMAGEN
# ---------------------------
def preprocess(img):
    img = ImageOps.grayscale(img)
    img = np.array(img)

    # Invertir (negro fondo, blanco dígito)
    img = cv2.bitwise_not(img)

    # Umbral para separar dígitos
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    return thresh


# ---------------------------
# SEGMENTAR DÍGITOS
# ---------------------------
def segment_digits(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    # Ordenar de izquierda a derecha
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filtrar ruido pequeño
        if w > 10 and h > 10:
            digit = img[y:y+h, x:x+w]
            digits.append(digit)

    return digits


# ---------------------------
# PREDECIR CADA DÍGITO
# ---------------------------
def predict_digit(digit_img):
    digit_img = cv2.resize(digit_img, (28, 28))
    digit_img = digit_img / 255.0
    digit_img = digit_img.reshape(1, 28, 28, 1)

    pred = model.predict(digit_img)
    return np.argmax(pred)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Reconocimiento múltiple", layout="centered")

st.title("🔢 Reconocimiento de hasta 3 dígitos")
st.markdown("Dibuja **1, 2 o 3 números juntos** ✍️")

stroke_width = st.slider("Grosor del trazo", 1, 30, 15)

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=400,   # MÁS ANCHO para varios dígitos
    key="canvas",
)

if st.button("✨ Predecir número"):
    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

        # Procesar imagen
        processed = preprocess(img)

        # Segmentar
        digits = segment_digits(processed)

        if len(digits) == 0:
            st.warning("No se detectaron números 😅")
        else:
            result = ""

            for d in digits[:3]:  # máximo 3
                pred = predict_digit(d)
                result += str(pred)

            st.success(f"🔢 Número detectado: {result}")

            # Mostrar cada dígito detectado
            cols = st.columns(len(digits[:3]))
            for i, d in enumerate(digits[:3]):
                cols[i].image(d, caption=f"Dígito {i+1}", width=80)

    else:
        st.warning("Dibuja algo primero ✍️")

# Sidebar
st.sidebar.title("ℹ️ Info")
st.sidebar.write("""
✔ Puedes escribir hasta **3 dígitos seguidos**  
✔ Ej: 123, 45, 789  
✔ Trata de separarlos un poco  
""")
