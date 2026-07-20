import gradio as gr
import requests


def chat(message):

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "message": message
        }
    )

    return response.json()["reply"]


demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask anything..."
    ),
    outputs=gr.Textbox(label="LLM Response"),
    title="Simple LLM Chat",
    description="Gradio + FastAPI + OpenAI",
    flagging_callback=None
)

demo.launch()