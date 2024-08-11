# STEP 1
from transformers import pipeline
from fastapi import FastAPI, Form

# STEP 2
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()

@app.post("/classification/")
async def login(text: str = Form()):
    # STEP 3
    # text

    # STEP 4
    result = classifier(text)

    # STEP 5
    return {"result": result}