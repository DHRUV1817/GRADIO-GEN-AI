import os
import random
import time
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Generator
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Research Paper Alignment: "Tree of Thoughts" & "Constitutional AI"
# Implements deliberate reasoning and self-correction mechanisms

class ReasoningMode(Enum):
    """Research-based reasoning approaches"""
    TREE_OF_THOUGHTS = "Tree of Thoughts (ToT)"
    CHAIN_OF_THOUGHT = "Chain of Thought (CoT)"
    SELF_CONSISTENCY = "Self-Consistency Sampling"
    REFLEXION = "Reflexion (Self-Reflection)"

@dataclass
class ResearchMetrics:
    """Track research-relevant performance metrics"""
    reasoning_depth: int = 0
    self_corrections: int = 0
    confidence_score: float = 0.0
    exploration_breadth: int = 0
    inference_time: float = 0.0

class TreeOfThoughtsReasoner:
    """
    Implementation based on: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    Yao et al., 2023
    
    Enables exploration of multiple reasoning paths before converging on solution
    """
    
    def __init__(self):
        self.reasoning_branches = []
        self.max_depth = 3
        
    def generate_reasoning_prompt(self, query: str, mode: ReasoningMode) -> str:
        """Generate research-aligned reasoning prompts"""
        
        prompts = {
            ReasoningMode.TREE_OF_THOUGHTS: f"""
Approach this problem using Tree of Thoughts methodology:

Problem: {query}

Generate 3 distinct reasoning paths:
PATH 1: [Analytical approach]
- Initial hypothesis:
- Key assumptions:
- Logical progression:

PATH 2: [Alternative perspective]
- Counter-arguments:
- Different framework:
- Validation checks:

PATH 3: [Synthesis]
- Integration of insights:
- Most robust solution:
- Confidence assessment:

Select and refine the most promising path.""",

            ReasoningMode.CHAIN_OF_THOUGHT: f"""
Solve step-by-step with explicit reasoning:

Problem: {query}

Step 1: Problem decomposition
Step 2: Identify key variables and constraints
Step 3: Apply relevant principles/theories
Step 4: Calculate or reason through solution
Step 5: Verify result and check edge cases

Final Answer: [Synthesized solution]""",

            ReasoningMode.SELF_CONSISTENCY: f"""
Generate multiple independent solutions and identify consensus:

Problem: {query}

Solution Attempt 1:
[Complete reasoning chain]

Solution Attempt 2:
[Alternative approach]

Solution Attempt 3:
[Third perspective]

Consensus Analysis: Compare solutions and identify most consistent answer.""",

            ReasoningMode.REFLEXION: f"""
Solve with self-reflection and error correction:

Problem: {query}

Initial Attempt:
[First solution]

Self-Critique:
- What assumptions did I make?
- Are there logical flaws?
- What did I overlook?

Refined Solution:
[Corrected approach based on reflection]

Final Validation:
[Quality checks and confidence score]"""
        }
        
        return prompts.get(mode, query)

class ConstitutionalAIFilter:
    """
    Based on: "Constitutional AI: Harmlessness from AI Feedback"
    Bai et al., 2022 (Anthropic)
    
    Implements self-critique and revision for safer, more accurate responses
    """
    
    def __init__(self):
        self.constitutional_principles = {
            "accuracy": "Prioritize factual correctness over speculation",
            "transparency": "Acknowledge uncertainty and limitations explicitly",
            "reasoning": "Show work and intermediate steps clearly",
            "safety": "Avoid harmful, biased, or misleading information"
        }
    
    def apply_constitutional_filter(self, response: str) -> str:
        """Add constitutional AI self-critique layer"""
        critique_prompt = f"""
Original Response: {response[:500]}...

Constitutional Self-Critique:
1. Accuracy Check: Are all factual claims verifiable?
2. Transparency: Have I acknowledged uncertainties?
3. Reasoning Quality: Is my logic clear and sound?
4. Safety: Could this response cause harm or perpetuate bias?

Provide refined response incorporating these checks."""
        return critique_prompt

class ResearchAISystem:
    """Main system implementing multiple research paper techniques"""
    
    def __init__(self):
        self.tot_reasoner = TreeOfThoughtsReasoner()
        self.constitutional_filter = ConstitutionalAIFilter()
        self.metrics = ResearchMetrics()
        
    def generate_research_response(
        self,
        query: str,
        history: List[tuple],
        model: str,
        reasoning_mode: ReasoningMode,
        enable_self_critique: bool,
        temperature: float,
        max_tokens: int
    ) -> Generator[str, None, None]:
        """Generate response using research-backed methodologies"""
        
        start_time = time.time()
        
        # Phase 1: Enhanced reasoning prompt
        enhanced_query = self.tot_reasoner.generate_reasoning_prompt(query, reasoning_mode)
        
        # Build conversation context
        messages = [
            {
                "role": "system",
                "content": """You are an advanced AI reasoning system implementing cutting-edge research methodologies:
- Tree of Thoughts for multi-path exploration
- Chain of Thought for transparent reasoning
- Self-Consistency for robust answers
- Reflexion for self-improvement

Always show your reasoning process explicitly. Prioritize accuracy over speed."""
            }
        ]
        
        # Add conversation history
        for user_msg, ai_msg in history[-5:]:  # Last 5 exchanges for context
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        
        messages.append({"role": "user", "content": enhanced_query})
        
        try:
            # Phase 2: Generate initial response
            stream = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield full_response
            
            # Phase 3: Constitutional AI self-critique (if enabled)
            if enable_self_critique and len(full_response) > 100:
                critique_prompt = self.constitutional_filter.apply_constitutional_filter(full_response)
                
                messages.append({"role": "assistant", "content": full_response})
                messages.append({"role": "user", "content": critique_prompt})
                
                refined_stream = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature * 0.8,  # Lower temp for critique
                    max_tokens=max_tokens // 2,
                    stream=True,
                )
                
                full_response += "\n\n---\n**🔍 Self-Critique & Refinement:**\n"
                for chunk in refined_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield full_response
                
                self.metrics.self_corrections += 1
            
            # Update metrics
            self.metrics.inference_time = time.time() - start_time
            self.metrics.reasoning_depth = full_response.count("Step") + full_response.count("PATH")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            yield f"⚠️ System Error: {str(e)}"

# Initialize system
research_system = ResearchAISystem()

# Modern UI Theme
custom_css = """
.research-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1rem;
}

.metrics-box {
    background: rgba(30, 60, 114, 0.1);
    border-left: 4px solid #2a5298;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}

.research-badge {
    background: #2a5298;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.85rem;
    display: inline-block;
    margin: 0.25rem;
}
"""

# Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css=custom_css,
    title="AI Reasoning Research System"
) as demo:
    
    gr.HTML("""
    <div class="research-header">
        <h1>🔬 Advanced AI Reasoning Research System</h1>
        <p><strong>Research Implementation:</strong> Tree of Thoughts + Constitutional AI + Multi-Path Reasoning</p>
        <div>
            <span class="research-badge">📄 Yao et al. 2023 - Tree of Thoughts</span>
            <span class="research-badge">📄 Bai et al. 2022 - Constitutional AI</span>
            <span class="research-badge">📄 Wei et al. 2022 - Chain of Thought</span>
        </div>
    </div>
    """)
    
    gr.Markdown("""
    ### 🎯 Research Focus
    This system addresses key challenges in AI reasoning:
    - **Problem:** Single-path reasoning limits solution quality
    - **Solution:** Multi-path exploration (Tree of Thoughts) + Self-correction (Constitutional AI)
    - **Impact:** More robust, transparent, and accurate problem-solving
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Research AI Interaction Space",
                height=500,
                show_copy_button=True,
                avatar_images=(
                    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                    "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
                )
            )
            
            msg = gr.Textbox(
                placeholder="Enter research question or complex problem...",
                label="Query Input",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("🚀 Process Query", variant="primary", size="lg")
                clear = gr.Button("🗑️ Reset", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Research Parameters")
            
            reasoning_mode = gr.Dropdown(
                choices=[mode.value for mode in ReasoningMode],
                value=ReasoningMode.TREE_OF_THOUGHTS.value,
                label="🧠 Reasoning Methodology",
                info="Select research-backed approach"
            )
            
            enable_critique = gr.Checkbox(
                label="🔍 Enable Constitutional AI Self-Critique",
                value=True,
                info="Applies Anthropic's Constitutional AI"
            )
            
            model = gr.Dropdown(
                choices=[
                    "llama-3.3-70b-versatile",
                    "deepseek-r1-distill-llama-70b",
                    "mixtral-8x7b-32768"
                ],
                value="llama-3.3-70b-versatile",
                label="🤖 Language Model"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(1000, 8000, value=4000, step=500, label="Max Tokens")
            
            gr.Markdown("### 📊 Research Metrics")
            
            with gr.Group():
                metrics_display = gr.Markdown("""
                <div class="metrics-box">
                <strong>Reasoning Depth:</strong> 0 steps<br>
                <strong>Self-Corrections:</strong> 0<br>
                <strong>Inference Time:</strong> 0.0s<br>
                <strong>Methodology:</strong> Not applied
                </div>
                """)
            
            gr.Markdown("""
            ### 📚 Research References
            - Yao et al. (2023) - Tree of Thoughts
            - Bai et al. (2022) - Constitutional AI
            - Wei et al. (2022) - Chain of Thought Prompting
            - Shinn et al. (2023) - Reflexion
            """)
    
    def process_query(message, history, mode, critique, model, temp, tokens):
        """Main query processing function"""
        
        reasoning_mode_enum = ReasoningMode(mode)
        
        full_response = ""
        for response in research_system.generate_research_response(
            message, history, model, reasoning_mode_enum, 
            critique, temp, tokens
        ):
            full_response = response
            yield history + [(message, full_response)], _update_metrics()
        
        yield history + [(message, full_response)], _update_metrics()
    
    def _update_metrics():
        """Update metrics display"""
        m = research_system.metrics
        return f"""
        <div class="metrics-box">
        <strong>Reasoning Depth:</strong> {m.reasoning_depth} steps<br>
        <strong>Self-Corrections:</strong> {m.self_corrections}<br>
        <strong>Inference Time:</strong> {m.inference_time:.2f}s<br>
        <strong>Status:</strong> Active
        </div>
        """
    
    # Event handlers
    msg.submit(
        process_query,
        [msg, chatbot, reasoning_mode, enable_critique, model, temperature, max_tokens],
        [chatbot, metrics_display]
    ).then(lambda: "", None, [msg])
    
    submit.click(
        process_query,
        [msg, chatbot, reasoning_mode, enable_critique, model, temperature, max_tokens],
        [chatbot, metrics_display]
    ).then(lambda: "", None, [msg])
    
    clear.click(lambda: (None, _update_metrics()), None, [chatbot, metrics_display])

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )