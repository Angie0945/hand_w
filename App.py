import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')

# Estilo personalizado para el contraste y centrado
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    /* Estilo para centrar el canvas y mejorar visibilidad */
    .stCanvas {
        margin: 0 auto;
        border: 2px solid #6f42c1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DEL MODELO ---
@st.cache_resource
def load_my_model():
    # Cargamos el modelo una sola vez para ahorrar memoria
    return tf.keras.models.load_model("model/handwritten.h5")

def predictDigit(image):
    model = load_my_model()
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img / 255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    return np.argmax(pred[0])

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("📌 Información")
    st.markdown("""
    En esta aplicación se evalúa la capacidad de una **Red Neuronal Artificial (RNA)** para reconocer dígitos escritos a mano.
    
    **Instrucciones:**
    1. Usa el mouse o panel táctil para dibujar.
    2. Ajusta el grosor si es necesario.
    3. Presiona el botón para procesar.
    
    ---
    *Basado en el desarrollo de Vinay Uniyal.*
    """)

# --- CUERPO PRINCIPAL ---
st.title('🖋️ Reconocimiento de Dígitos')
st.subheader("Dibuja un dígito en el panel central")

# Columnas para centrar el contenido
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Configuración de colores solicitada
    bg_color = "#E0BBE4"  # Color Lila
    stroke_color = "#000000" # Letra negra para contraste
    
    stroke_width = st.slider('Ajustar grosor de línea', 1, 30, 15)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280, # Tamaño aumentado para mejor manejo
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button('✨ Predecir ahora', use_container_width=True):
        if canvas_result.image_data is not None:
            # Procesamiento de imagen
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            
            # Nota: El modelo suele esperar fondo negro y letra blanca (MNIST)
            # Si dibujas negro sobre lila, debemos invertir o procesar adecuadamente
            res = predictDigit(input_image)
            
            st.success(f'### 🎯 Resultado: {res}')
        else:
            st.warning('Por favor, dibuja algo en el panel antes de predecir.')
