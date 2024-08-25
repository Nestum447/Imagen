import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Título de la aplicación
st.title("Aplicación de Estilo Artístico")

# Opción para cargar la imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar la imagen
    image = Image.open(uploaded_file)

    # Mostrar la imagen original
    st.image(image, caption="Imagen original", use_column_width=True)

    # Preprocesar la imagen
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Crear un batch de una imagen

    # Cargar el modelo preentrenado
    model = models.vgg19(pretrained=True).features.eval()

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Pasar la imagen a través del modelo
    with torch.no_grad():
        output = model(input_batch)

    # Convertir la salida de vuelta a una imagen
    unloader = transforms.ToPILImage()
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
