from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = OpenAI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):

    response = client.responses.create(
        model="gpt-4o-mini",
        input=request.message
    )

    return {
        "reply": response.output_text
    }