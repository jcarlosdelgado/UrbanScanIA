from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from typing import List
from fastapi.middleware.cors import CORSMiddleware # Importar CORSMiddleware

# Cargar procesador y modelo
processor = AutoImageProcessor.from_pretrained("cm93/resnet50-eurosat")
model = AutoModelForImageClassification.from_pretrained("cm93/resnet50-eurosat")
model.eval()
labels = model.config.id2label
urban_labels = ['residential', 'industrial', 'highway']

app = FastAPI()

# Configuración de CORS
# Define los orígenes que están permitidos para hacer solicitudes a tu API.
# Durante el desarrollo, puedes usar ["*"] para permitir todos los orígenes.
# Para producción, es ALTAMENTE RECOMENDABLE especificar los orígenes exactos
# de tu frontend de Angular (por ejemplo, ["http://localhost:4200", "https://tudominio.com"]).
origins = [
    "http://localhost",
    "http://localhost:8080", # Ejemplo de otro origen local si lo necesitas
    "http://localhost:4200", # Asumiendo que tu frontend de Angular corre en este puerto por defecto
    "*" # Permite todos los orígenes - ÚTIL SOLO PARA DESARROLLO.
        # ¡Elimina o restringe esto para entornos de producción!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Lista de orígenes permitidos
    allow_credentials=True, # Permite cookies (si tu app usa sesiones/autenticación con cookies)
    allow_methods=["*"], # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Permite todos los encabezados HTTP
)

@app.post("/clasificar_imagen")
async def clasificar_imagen_api(file: UploadFile = File(...)):
    try:
        # Verificar si el archivo es una imagen
        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Por favor, sube un archivo de imagen."}, status_code=400)

        # Leer la imagen
        image = Image.open(file.file).convert("RGB")

        # Procesar la imagen
        inputs = processor(images=image, return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            etiqueta = labels[pred_idx]
            probabilidad = probs[0][pred_idx].item()

        # Verificar si la zona está urbanizada
        es_urbanizado = etiqueta.lower() in urban_labels

        # Devolver los resultados en formato JSON
        return JSONResponse(content={
            "etiqueta_predicha": etiqueta,
            "probabilidad": f"{probabilidad:.2f}",
            "es_urbanizado": es_urbanizado
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
#Primero crear el entorno virtual que es el venv
#venv\Scripts\activate
#uvicorn main:app --reload
#pip install torch torchvision transformers pillow timm uvicorn fastapi python-multipart

