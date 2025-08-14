import uvicorn
import os

if __name__ == "__main__":
    # Puedes configurar el host y puerto aquí
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True  # Para desarrollo, quítalo en producción
    )
