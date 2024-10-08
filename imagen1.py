import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import models, transforms
import numpy as np

# Título de la aplicación
st.title("Aplicación de Estilo Artístico")

# Opción para cargar la imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar la imagen
    image = Image.open(uploaded_file)

    # Ajustes de brillo y contraste
    brightness = st.slider('Brillo', 0.0, 2.0, 1.0)
    contrast = st.slider('Contraste', 0.0, 2.0, 1.0)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Opción para aplicar un filtro
    filter_option = st.selectbox('Selecciona un filtro', ['Ninguno', 'Desenfoque', 'Bordes'])
    if filter_option == 'Desenfoque':
        image = image.filter(ImageFilter.BLUR)
    elif filter_option == 'Bordes':
        image = image.filter(ImageFilter.FIND_EDGES)

    st.image(image, caption="Imagen ajustada", use_column_width=True)

    # Preprocesar la imagen
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Crear un batch de una imagen

    # Determinar si usar CPU o GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el modelo preentrenado
    model = models.vgg19(pretrained=True).features.to(device).eval()

    # Mover el batch de la imagen al dispositivo
    input_batch = input_batch.to(device)

    # Pasar la imagen a través del modelo
    with torch.no_grad():
        output = model(input_batch)

    # Convertir la salida de vuelta a una imagen
    unloader = transforms.ToPILImage()
    output_shape = output.squeeze(0).shape
    if output_shape[0] > 3:
        output = output[:, :3, :, :]
    image_output = unloader(output.squeeze(0).cpu())

    # Mostrar la imagen procesada
    st.image(image_output, caption="Imagen con estilo aplicado", use_column_width=True)

    # Opción para descargar la imagen procesada
    st.download_button(
        label="Descargar imagen",
        data=image_output.tobytes(),
        file_name="output_stylized.jpg",
        mime="image/jpeg"
    )
