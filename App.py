import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------------- CONFIG ----------------
st.set_page_config(page_title='Reconocimiento de Números', layout='centered')

# ---------------- ESTILO ----------------
st.markdown("""
<style>
.stApp {
    background-color: #E6D9FF;  /* lila suave */
}

.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
}

canvas {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🔢 Reconocimiento de Números</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Dibuja números (pueden estar juntos)</p>', unsafe_allow_html=True)

# ---------------- CARGAR MODELO ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# ---------------- FUNCIÓN PREDICCIÓN ----------------
def predict_digit(img):
    img = img.resize((28, 28))
    img = ImageOps.grayscale(img)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img, verbose=0)
    return np.argmax(pred)

# ---------------- SEGMENTAR VARIOS DÍGITOS ----------------
def segment_digits(image):
    img = ImageOps.grayscale(image)
    img = np.array(img)

    # binarizar
    img = (img > 50).astype(np.uint8) * 255

    # sumar columnas
    col_sum = np.sum(img, axis=0)

    digits = []
    start = None

    for i, val in enumerate(col_sum):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > 5:
                digit = img[:, start:i]
                digits.append(digit)
            start = None

    return digits

# ---------------- CANVAS CENTRADO ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    stroke_width = st.slider('🖊️ Grosor', 5, 40, 20)

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# ---------------- PREDICCIÓN ----------------
st.markdown("###")

if st.button("✨ Predecir número"):

    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert("RGB")

        digits = segment_digits(img)

        if len(digits) == 0:
            st.warning("⚠️ No se detectaron números")
        else:
            result = ""

            for d in digits:
                pil_img = Image.fromarray(d)
                num = predict_digit(pil_img)
                result += str(num)

            st.success(f"🔢 Número detectado: {result}")

    else:
        st.warning("⚠️ Dibuja algo primero")

# ---------------- INFO ----------------
st.markdown("---")
st.caption("💜 Dibuja números claros. Puedes escribir varios juntos.")
