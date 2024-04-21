import os
import gradio as gr
from groq import Groq

import gradio as gr
from groq import Groq

def generate_response(input_text, model, temperature, max_tokens, top_p):
    client = Groq()

    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=None,
        stream=True,
    )

    response = ""
    for chunk in stream:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            response += delta_content

    return response

# Define the Gradio chat interface
additional_inputs = [
    gr.Dropdown(choices=["mixtral-8x7b-32768", "mixtral-12x7b-32768"], label="Model"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Temperature"),
    gr.Slider(minimum=1, maximum=4096, step=1, label="Max Tokens"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Top P"),
]

gr.ChatInterface(
    fn=generate_response,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="Groq API LLMs AI Models",
    description="Using https://groq.com/ api, ofc as its free it will have some limitations so its better if you duplicate this space with your own api key<br>Hugging Face Space by [Nick088](https://linktr.ee/Nick088",
).launch()