from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import time

# Initialize FastAPI
app = FastAPI(
    title="UrbanScanIA API",
    description="AI-powered analysis of urban satellite imagery",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "cm93/resnet50-eurosat"
print(f"Loading model: {MODEL_NAME}...")

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    labels = model.config.id2label
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit if model fails to load
    raise RuntimeError(f"Could not load ML model: {e}")

# Detailed descriptions for EuroSAT classes
CLASS_INSIGHTS = {
    "annualcrop": {
        "description": "Zonas de cultivos anuales. Alta productividad agrícola detectada.",
        "recommendation": "Fomentar prácticas de rotación de cultivos para mantener la salud del suelo.",
        "sustainability_score": 85
    },
    "forest": {
        "description": "Área forestal densa. Pulmón ecológico crítico.",
        "recommendation": "Implementar zonas de protección estricta contra la deforestación.",
        "sustainability_score": 98
    },
    "herbaceousvegetation": {
        "description": "Vegetación herbácea natural. Ecosistema de pastizales o praderas.",
        "recommendation": "Monitorear la biodiversidad local y prevenir el sobrepastoreo.",
        "sustainability_score": 92
    },
    "highway": {
        "description": "Infraestructura de transporte mayor. Conectividad vial detectada.",
        "recommendation": "Evaluar la integración de barreras acústicas y corredores biológicos.",
        "sustainability_score": 35
    },
    "industrial": {
        "description": "Zona industrial pesada. Alta huella de carbono potencial.",
        "recommendation": "Implementar monitoreo de emisiones y transición a energías verdes.",
        "sustainability_score": 20
    },
    "pasture": {
        "description": "Zonas de pastoreo. Uso de suelo agrícola-ganadero.",
        "recommendation": "Optimizar el uso de agua y reducir el impacto del metano.",
        "sustainability_score": 75
    },
    "permanentcrop": {
        "description": "Cultivos permanentes (frutales, viñedos). Estabilidad agrícola.",
        "recommendation": "Mejorar sistemas de riego eficiente por goteo.",
        "sustainability_score": 80
    },
    "residential": {
        "description": "Asentamientos urbanos residenciales. Alta densidad poblacional.",
        "recommendation": "Aumentar el índice de áreas verdes por habitante.",
        "sustainability_score": 55
    },
    "river": {
        "description": "Cuerpo de agua corriente. Recurso hídrico vital.",
        "recommendation": "Monitorear la calidad del agua y restaurar riberas naturales.",
        "sustainability_score": 95
    },
    "sealake": {
        "description": "Masas de agua estables (mar o lagos). Ecosistema acuático.",
        "recommendation": "Prevenir vertidos industriales y proteger la fauna marina.",
        "sustainability_score": 90
    }
}

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "UrbanScanIA Advanced Intelligence Platform",
        "endpoints": ["/analyze", "/health", "/metadata"],
        "version": "2.0.0"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ML Inference
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
            label = labels[pred_idx]
            confidence = probs[0][pred_idx].item()

        # Business logic: Is it urban?
        is_urbanized = label.lower() in URBAN_LABELS
        
        # Enhanced Insights
        insight = CLASS_INSIGHTS.get(label.lower(), {
            "description": "Clase no documentada en el sistema experto.",
            "recommendation": "Consultar manual técnico de clasificación.",
            "sustainability_score": 50
        })
        
        analysis_time = time.time() - start_time
        
        return {
            "success": True,
            "data": {
                "prediction": label,
                "confidence": round(confidence * 100, 2),
                "is_urbanized": is_urbanized,
                "category": "Urban" if is_urbanized else "Natural/Rural",
                "description": insight["description"],
                "recommendation": insight["recommendation"],
                "sustainability_index": insight["sustainability_score"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "metadata": {
                "analysis_time_ms": round(analysis_time * 1000, 2),
                "model": MODEL_NAME,
                "inference_engine": "PyTorch/Transformers"
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

