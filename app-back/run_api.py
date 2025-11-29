import uvicorn
import os

if __name__ == "__main__":
    # You can configure host and port via environment variables
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True  # Enable auto-reload for development; disable in production
    )
