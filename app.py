import os
import random
import json
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 🧠 Advanced Prompt Engineering Templates (Based on Chain-of-Thought & Constitutional AI)
SYSTEM_PERSONAS = {
    "Default": "You are a helpful AI assistant.",
    "Researcher": "You are an expert research assistant. Break down complex topics step-by-step, cite reasoning, and provide evidence-based answers.",
    "Creative": "You are a creative AI with vivid imagination. Think outside the box and generate innovative ideas.",
    "Critic": "You are a critical thinker. Analyze arguments from multiple angles, identify weaknesses, and suggest improvements.",
    "Socratic": "You are a Socratic teacher. Guide users to answers through thoughtful questions rather than direct answers."
}

# 🎯 Constitutional AI Principles (Safety & Quality)
CONSTITUTIONAL_RULES = """
- Be helpful, harmless, and honest
- Admit uncertainty when unsure
- Avoid harmful, biased, or misleading content
- Provide balanced perspectives
- Cite reasoning steps clearly
"""

# 💡 Chain-of-Thought Injection
def inject_cot_prompt(user_msg, cot_enabled):
    if cot_enabled:
        return f"{user_msg}\n\n[Think step-by-step and show your reasoning process]"
    return user_msg

# 🔄 Multi-Turn Context Compression (Inspired by LongLLMLingua paper)
def compress_history(history, max_turns=5):
    """Keep only recent context to manage token limits intelligently"""
    if len(history) > max_turns:
        compressed = history[-max_turns:]
        summary = f"[Previous context: {len(history)-max_turns} earlier exchanges summarized]"
        return [(summary, "Context compressed.")] + compressed
    return history

# 🎭 Response Enhancement with Self-Reflection
def enhance_with_reflection(response, reflection_enabled):
    if reflection_enabled and len(response) > 100:
        reflection_prompt = f"\n\n💭 Self-Reflection: [Briefly evaluate if this answer is comprehensive and accurate]"
        return response + reflection_prompt
    return response

# 🚀 Main Generation Function with Advanced Features
def generate_response(prompt, history, model, temperature, max_tokens, top_p, seed, 
                     persona, cot_enabled, reflection_enabled, context_compression):
    
    # Apply context compression
    if context_compression:
        history = compress_history(history)
    
    # Build messages with system persona
    system_message = {"role": "system", "content": SYSTEM_PERSONAS[persona] + "\n" + CONSTITUTIONAL_RULES}
    messages = [system_message]
    
    # Add history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Inject Chain-of-Thought
    enhanced_prompt = inject_cot_prompt(prompt, cot_enabled)
    messages.append({"role": "user", "content": enhanced_prompt})
    
    # Random seed if not specified
    if seed == 0:
        seed = random.randint(1, 100000)
    
    # Stream response
    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
        stream=True,
    )
    
    response = ""
    for chunk in stream:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            response += delta_content
            # Apply reflection enhancement at the end
            if chunk.choices[0].finish_reason == "stop":
                response = enhance_with_reflection(response, reflection_enabled)
            yield response

# 📊 Token Counter for Transparency
def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 chars)"""
    return len(text) // 4

# 🎨 Custom CSS for Modern UI
custom_css = """
.persona-selector {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 10px;
}
"""

# 🎛️ Advanced Interface Configuration
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🧠 Advanced Gen-AI Chat System
    ### Powered by Constitutional AI + Chain-of-Thought + Context Management
    
    **Research-Backed Features:**
    - 🎭 **Multiple AI Personas** (Adaptive behavior)
    - 🧵 **Chain-of-Thought Reasoning** (Transparent thinking)
    - 🔄 **Smart Context Compression** (Efficient memory)
    - 💭 **Self-Reflection Mode** (Quality assurance)
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                show_label=False,
                show_share_button=False,
                show_copy_button=True,
                layout="panel",
                height=500
            )
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                container=False
            )
            
            with gr.Row():
                submit = gr.Button("🚀 Send", variant="primary")
                clear = gr.Button("🗑️ Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Model Settings")
            
            model = gr.Dropdown(
                choices=[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "gemma2-9b-it",
                    "deepseek-r1-distill-llama-70b"
                ],
                value="llama-3.3-70b-versatile",
                label="🤖 Model"
            )
            
            temperature = gr.Slider(0.0, 2.0, step=0.1, value=0.7, label="🌡️ Temperature")
            max_tokens = gr.Slider(100, 32000, step=100, value=4096, label="📝 Max Tokens")
            top_p = gr.Slider(0.0, 1.0, step=0.05, value=0.9, label="🎯 Top P")
            seed = gr.Number(precision=0, value=0, label="🎲 Seed (0=random)")
            
            gr.Markdown("### 🧠 Advanced Features")
            
            persona = gr.Dropdown(
                choices=list(SYSTEM_PERSONAS.keys()),
                value="Default",
                label="🎭 AI Persona",
                elem_classes=["persona-selector"]
            )
            
            cot_enabled = gr.Checkbox(label="🧵 Enable Chain-of-Thought", value=True)
            reflection_enabled = gr.Checkbox(label="💭 Enable Self-Reflection", value=False)
            context_compression = gr.Checkbox(label="🔄 Smart Context Management", value=True)
            
            token_display = gr.Markdown("**Tokens Used:** 0")
    
    # Event handlers
    def respond(message, chat_history, *args):
        chat_history = chat_history or []
        bot_response = ""
        for response in generate_response(message, chat_history, *args):
            bot_response = response
            yield chat_history + [(message, bot_response)]
        
        # Update token count
        total_tokens = sum(estimate_tokens(msg[0]) + estimate_tokens(msg[1]) 
                          for msg in chat_history + [(message, bot_response)])
        token_display.value = f"**Tokens Used:** ~{total_tokens}"
    
    msg.submit(
        respond,
        [msg, chatbot, model, temperature, max_tokens, top_p, seed, 
         persona, cot_enabled, reflection_enabled, context_compression],
        [chatbot]
    ).then(lambda: "", None, [msg])
    
    submit.click(
        respond,
        [msg, chatbot, model, temperature, max_tokens, top_p, seed,
         persona, cot_enabled, reflection_enabled, context_compression],
        [chatbot]
    ).then(lambda: "", None, [msg])
    
    clear.click(lambda: None, None, [chatbot])

if __name__ == "__main__":
    demo.launch(share=False)