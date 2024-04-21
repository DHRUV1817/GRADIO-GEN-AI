import subprocess
import os
import gradio as gr
from groq import Groq

groq_api_key = os.environ.get('Groq_Api_Key')

subprocess.run(["export", f"GROQ_API_KEY={groq_api_key}"], check=True)

def generate_response(input_text):
    client = Groq()

    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": input_text}
        ],
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
    )

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content

    return response

# Define the Gradio UI
inputs = gr.Textbox(label="Enter your question")
outputs = gr.Textbox(label="Model Response")

gr.Interface(
    fn=generate_response,
    inputs=inputs,
    outputs=outputs,
    title="Language Model Assistant",
    description="Ask questions and get responses from a language model.",
).launch(show_api=False, share=True)