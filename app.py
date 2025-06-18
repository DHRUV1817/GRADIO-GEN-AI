import os
import random
import gradio as gr
from groq import Groq

# Initialize the Groq client with your API key
client = Groq(
    api_key=os.environ.get("Groq_Api_Key")
)

def create_history_messages(history):
    # Interleave user and assistant messages in the order they occurred
    history_messages = []
    for user_msg, assistant_msg in history:
        history_messages.append({"role": "user", "content": user_msg})
        history_messages.append({"role": "assistant", "content": assistant_msg})
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

additional_inputs = [
    gr.Dropdown(
        choices=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "deepseek-r1-distill-llama-70b"
        ],
        value="llama-3.1-70b-versatile",
        label="Model"
    ),
    gr.Slider(
        minimum=0.0, maximum=1.0, step=0.01, value=0.5,
        label="Temperature",
        info="Controls diversity of the generated text. Lower is more deterministic, higher is more creative."
    ),
    gr.Slider(
        minimum=1, maximum=131000, step=1, value=8100,
        label="Max Tokens",
        info="The maximum number of tokens that the model can process in a single response.<br>Maximums: 8k for gemma 7b it, gemma2 9b it, llama 7b & 70b, 32k for mixtral 8x7b, 132k for llama 3.1."
    ),
    gr.Slider(
        minimum=0.0, maximum=1.0, step=0.01, value=0.5,
        label="Top P",
        info="A method of text generation where a model will only consider the most probable next tokens that make up the probability p."
    ),
    gr.Number(
        precision=0, value=0, label="Seed",
        info="A starting point to initiate generation, use 0 for random"
    )
]

gr.ChatInterface(
    fn=generate_response,
    theme="Nymbo/Alyx_Theme",
    chatbot=gr.Chatbot(
        show_label=False,
        show_share_button=False,
        show_copy_button=True,
        likeable=True,
        layout="panel"
    ),
    additional_inputs=additional_inputs,
).launch()