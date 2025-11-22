from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FastAPI Application", version="1.0.0")

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/dumy-histo")
async def get_dummy_histogram():
    dummy_histogram = {
            "0-10"  : 5,
            "10-20" : 15,
            "20-30" : 25,
            "30-40" : 10,
            "40-50" : 8,
            "50-60" : 12,
            "60-70" : 7,
            "70-80" : 3,
            "80-90" : 1,
            "90-100": 0,
    }
    return dummy_histogram
