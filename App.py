import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------------------------
# CONFIG UI BONITA
# ---------------------------
st.set_page_config(page_title="Reconocimiento de Dígitos", layout="centered")

st.markdown("""
<style>
body {
    background-color: #F5F7FB;
}
.big-title {
    font-size: 36px;
    text-align: center;
    font-weight: bold;
}
.stButton>button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🔢 Reconocimiento de Dígitos</p>', unsafe_allow_html=True)
st.write("Dibuja un número ✍️ y presiona **Predecir**")

# ---------------------------
# CARGAR MODELO
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# ---------------------------
# PREPROCESAMIENTO CORRECTO
# ---------------------------
def preprocess(image):
    image = ImageOps.grayscale(image)

    # Invertir colores (fondo negro → blanco)
    image = ImageOps.invert(image)

    # Redimensionar
    image = image.resize((28, 28))

    img = np.array(image)

    # Normalizar
    img = img / 255.0

    # Ajustar forma
    img = img.reshape(1, 28, 28, 1)

    return img

# ---------------------------
# CANVAS (MEJOR VISUAL)
# ---------------------------
stroke_width = st.slider("✏️ Grosor del trazo", 5, 25, 15)

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000000",   # negro
    background_color="#FFFFFF",  # fondo blanco (MEJOR)
    height=250,
    width=250,
    key="canvas",
)

# ---------------------------
# BOTONES BONITOS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predecir")

with col2:
    clear_btn = st.button("🧹 Limpiar")

if clear_btn:
    st.rerun()

# ---------------------------
# PREDICCIÓN
# ---------------------------
if predict_btn:
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
# SIDEBAR LIMPIO
# ---------------------------
st.sidebar.title("ℹ️ Tips")
st.sidebar.write("""
✔ Dibuja un solo número  
✔ Hazlo grande y centrado  
✔ Evita trazos muy finos  
✔ No lo pegues a los bordes  
""")
