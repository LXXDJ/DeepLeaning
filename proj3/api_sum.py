# STEP 1
from transformers import pipeline
from fastapi import FastAPI, Form

# STEP 2
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

app = FastAPI()

@app.post("/summarization/")
async def login(text: str = Form()):

    # STEP 4
    result = summarizer(text)

    # STEP 5
    return {"result": result}