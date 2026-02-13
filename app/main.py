"""FastAPI application entry point"""

from fastapi import FastAPI

app = FastAPI(title="FastAPI Application")

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)