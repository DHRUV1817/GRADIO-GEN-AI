import os
import random
import gradio as gr
from groq import Groq

def generate_response(prompt, history, model, temperature, max_tokens, top_p, seed):
    client = Groq(
        api_key = os.environ.get("Groq_Api_Key")
    )

    if seed == 0:
        seed = random.randint(1, 100000)

    input_text = prompt + history
    
    stream = client.chat.completions.create(
        messages=input_text,
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

# Define the Gradio chat interface
additional_inputs = [
    gr.Dropdown(choices=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"], value="llama3-70b-8192", label="LLM Model"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Temperature", info="Controls randomness of responses"),
    gr.Slider(minimum=1, maximum=4096, step=1, value=4096, label="Max Tokens", info="The maximum number of tokens that the model can process in a single response"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Top P", info="A method of text generation where a model will only consider the most probable next tokens that make up the probability p."),
    gr.Number(precision=0, value=42, label="Seed", info="A starting point to initiate generation, use 0 for random")
]

gr.ChatInterface(
    fn=generate_response,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="Groq API LLMs AI Models",
    description="Using https://groq.com/ api, ofc as its free it will have some limitations of requests per minute, so its better if you duplicate this space with your own api key<br>Hugging Face Space by [Nick088](https://linktr.ee/Nick088)",
).launch()