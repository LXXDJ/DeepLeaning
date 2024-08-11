# STEP 1
from transformers import pipeline
from fastapi import FastAPI, Form

# STEP 2
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

app = FastAPI()


@app.post("/question_anwerer/")
async def login(question: str = Form(), context: str = Form()):
    # STEP 4
    result = question_answerer(question=question, context=context)
    
    # STEP 5
    return {"result": result}