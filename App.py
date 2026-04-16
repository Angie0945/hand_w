import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Reconocimiento de Dígitos", layout="centered")

# ---------------------------
# ESTILO (CLAVE PARA ICONOS)
# ---------------------------
st.markdown("""
<style>
/* Fondo general */
body {
    background-color: #F5F7FB;
}

/* Título */
.big-title {
    font-size: 36px;
    text-align: center;
    font-weight: bold;
}

/* Botones principales */
.stButton>button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}

/* 👇 ESTO ARREGLA LOS ICONOS DEL CANVAS */
button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
}

canvas {
    border: 2px solid #ddd;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# UI
# ---------------------------
st.markdown('<p class="big-title">🔢 Reconocimiento de Dígitos</p>', unsafe_allow_html=True)
st.write("Dibuja un número ✍️ y presiona **Predecir**")

# ---------------------------
# MODELO
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# ---------------------------
# PREPROCESAMIENTO
# ---------------------------
def preprocess(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)  # importante
    image = image.resize((28, 28))

    img = np.array(image) / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# ---------------------------
# CANVAS (SIN BOTÓN LIMPIAR)
# ---------------------------
stroke_width = st.slider("✏️ Grosor del trazo", 5, 25, 15)

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=250,
    width=250,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------
# BOTÓN PREDICT
# ---------------------------
if st.button("🔍 Predecir"):
    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

        processed = preprocess(img)

        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"🎯 Resultado: **{digit}**")
        st.write(f"Confianza: {confidence:.2f}")

    else:
        st.warning("Dibuja un número primero ✍️")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("💡 Tips")
st.sidebar.write("""
✔ Usa los botones debajo del canvas  
✔ (deshacer, borrar, etc.)  
✔ Dibuja centrado  
✔ Evita trazos muy finos  
""")
