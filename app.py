import os
import random
import gradio as gr
from groq import Groq

client = Groq(
    api_key = os.environ.get("Groq_Api_Key")
)
    
def create_history_messages(history):
    history_messages = [{"role": "user", "content": m[0]} for m in history]
    history_messages.extend([{"role": "assistant", "content": m[1]} for m in history])
    return history_messages

def generate_response(prompt, history, model, temperature, max_tokens, top_p, seed):
    messages = create_history_messages(history)
    messages.append({"role": "user", "content": prompt})
    print(messages)

    if seed == 0:
        seed = random.randint(1, 100000)
    
    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
        stop=None,
        stream=True,
    )

    response = ""
    for chunk in stream:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            response += delta_content
            yield response

    return response

additional_inputs = [
    gr.Dropdown(choices=["llama-3.1-405b-reasoning", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it"], value="llama-3.1-405b-reasoning", label="Model"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Temperature", info="Controls diversity of the generated text. Lower is more deterministic, higher is more creative."),
    gr.Slider(minimum=1, maximum=131000, step=1, value=8100, label="Max Tokens", info="The maximum number of tokens that the model can process in a single response.<br>Maximums: 8k for gemma 7b it, gemma2 9b it, llama 7b & 70b, 32k for mixtral 8x7b, 132k for llama 3.1."),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Top P", info="A method of text generation where a model will only consider the most probable next tokens that make up the probability p."),
    gr.Number(precision=0, value=0, label="Seed", info="A starting point to initiate generation, use 0 for random")
]

gr.ChatInterface(
    fn=generate_response, theme="Nymbo/Alyx_Theme",
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
).launch()