# ruta_imagen = "C:/Users/Darka/Documents/tensorflow api/imagenSatelitalSantaCruz.png"

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests

# Cargar procesador y modelo
processor = AutoImageProcessor.from_pretrained("cm93/resnet50-eurosat")
model = AutoModelForImageClassification.from_pretrained("cm93/resnet50-eurosat")
model.eval()  # Modo evaluación

# Diccionario de etiquetas
labels = model.config.id2label

def clasificar_imagen(ruta_imagen):
    # Cargar imagen y convertir a RGB
    image = Image.open(ruta_imagen).convert("RGB")

    # Procesar imagen
    inputs = processor(images=image, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        print(outputs.__dict__)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        etiqueta = labels[pred_idx]
        probabilidad = probs[0][pred_idx].item()

    # Verificar si la zona está urbanizada
    urban_labels = ['residential', 'industrial', 'highway']
    es_urbanizado = etiqueta.lower() in urban_labels

    # Resultados
    print(f"Etiqueta predicha: {etiqueta}")
    print(f"Probabilidad: {probabilidad:.2f}")
    print(f"¿Zona urbanizada?: {'Sí' if es_urbanizado else 'No'}")

# Ruta a la imagen satelital
ruta_imagen = "C:/Users/Darka/Documents/tensorflow api/imagenSatelitalSantaCruz.png"
clasificar_imagen(ruta_imagen)


