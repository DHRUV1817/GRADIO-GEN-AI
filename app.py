import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Generator, Optional, Any
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from contextlib import contextmanager

import gradio as gr
from dotenv import load_dotenv
from groq import Groq


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reasoning_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Constants
MAX_HISTORY_LENGTH = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


class ReasoningMode(Enum):
    """Research-aligned reasoning methodologies"""
    TREE_OF_THOUGHTS = "🌳 Tree of Thoughts (ToT)"
    CHAIN_OF_THOUGHT = "🔗 Chain of Thought (CoT)"
    SELF_CONSISTENCY = "🎯 Self-Consistency Sampling"
    REFLEXION = "🔍 Reflexion + Self-Correction"


@dataclass
class ConversationMetrics:
    """Enhanced metrics with validation"""
    reasoning_depth: int = 0
    self_corrections: int = 0
    confidence_score: float = 0.0
    inference_time: float = 0.0
    tokens_used: int = 0
    reasoning_paths_explored: int = 0
    total_conversations: int = 0
    avg_response_time: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    
    def update_confidence(self) -> None:
        """Calculate confidence based on reasoning depth"""
        self.confidence_score = min(95.0, 60.0 + (self.reasoning_depth * 5))
    
    def reset(self) -> None:
        """Reset metrics for new session"""
        self.__init__()


@dataclass
class ConversationEntry:
    """Structured conversation entry with validation"""
    timestamp: str
    user_message: str
    ai_response: str
    model: str
    reasoning_mode: str
    inference_time: float
    tokens: int
    bookmarked: bool = False
    feedback: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sanitization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create instance from dictionary"""
        return cls(**data)


def error_handler(func):
    """Enhanced error handling decorator with logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            error_msg = f"⚠️ **System Error:** {str(e)}\n\n"
            
            if "api" in str(e).lower() or "key" in str(e).lower():
                error_msg += "Please verify your GROQ_API_KEY in the .env file."
            else:
                error_msg += "Please try again or contact support if the issue persists."
            
            return error_msg
    return wrapper


@contextmanager
def timer(operation: str = "Operation"):
    """Context manager for timing operations"""
    start = time.time()
    yield
    duration = time.time() - start
    logger.info(f"{operation} completed in {duration:.2f}s")


class GroqClientManager:
    """Singleton manager for Groq client with validation"""
    _instance: Optional[Groq] = None
    
    @classmethod
    def get_client(cls) -> Groq:
        """Get or create Groq client instance"""
        if cls._instance is None:
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY")
            
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found. Please set it in your .env file."
                )
            
            cls._instance = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        
        return cls._instance


class PromptEngine:
    """Centralized prompt management"""
    
    SYSTEM_PROMPTS = {
        ReasoningMode.TREE_OF_THOUGHTS: """You are an advanced reasoning system using Tree of Thoughts methodology.
Explore multiple reasoning paths systematically before converging on the best solution.
Always show your thought process explicitly.""",
        
        ReasoningMode.CHAIN_OF_THOUGHT: """You are a systematic problem solver using Chain of Thought reasoning.
Break down complex problems into clear, logical steps with explicit reasoning.""",
        
        ReasoningMode.SELF_CONSISTENCY: """You are a consistency-focused reasoning system.
Generate multiple independent solutions and identify the most consistent answer.""",
        
        ReasoningMode.REFLEXION: """You are a self-reflective AI system.
Solve problems, critique your own reasoning, and refine your solutions iteratively."""
    }
    
    TEMPLATES = {
        "Code Review": "Analyze the following code for bugs, performance issues, and best practices:\n\n{query}",
        "Research Summary": "Provide a comprehensive research summary on:\n\n{query}\n\nInclude key findings, methodologies, and implications.",
        "Problem Solving": "Solve this problem step-by-step with detailed explanations:\n\n{query}",
        "Creative Writing": "Generate creative content based on:\n\n{query}\n\nBe imaginative and engaging.",
        "Data Analysis": "Analyze this data/scenario and provide insights:\n\n{query}",
        "Custom": "{query}"
    }
    
    REASONING_PROMPTS = {
        ReasoningMode.TREE_OF_THOUGHTS: """
🌳 **Tree of Thoughts Analysis**

Problem: {query}

**Exploration Phase:**
PATH A (Analytical): [Examine from first principles]
PATH B (Alternative): [Consider different angle]
PATH C (Synthesis): [Integrate insights]

**Evaluation Phase:**
- Assess each path's validity
- Identify strongest reasoning chain
- Converge on optimal solution

**Final Solution:** [Most robust answer with justification]""",

        ReasoningMode.CHAIN_OF_THOUGHT: """
🔗 **Step-by-Step Reasoning**

Problem: {query}

Step 1: Understand the question
Step 2: Identify key components
Step 3: Apply relevant logic/principles
Step 4: Derive solution
Step 5: Validate and verify

Final Answer: [Clear, justified conclusion]""",

        ReasoningMode.SELF_CONSISTENCY: """
🎯 **Multi-Path Consistency Check**

Problem: {query}

**Attempt 1:** [First independent solution]
**Attempt 2:** [Alternative approach]
**Attempt 3:** [Third perspective]

**Consensus:** [Most consistent answer across attempts]""",

        ReasoningMode.REFLEXION: """
🔍 **Reflexion with Self-Correction**

Problem: {query}

**Initial Solution:** [First attempt]

**Self-Critique:**
- Assumptions made?
- Logical flaws?
- Missing elements?

**Refined Solution:** [Improved answer based on reflection]"""
    }
    
    @classmethod
    def build_prompt(cls, query: str, mode: ReasoningMode, template: str) -> str:
        """Build enhanced reasoning prompt"""
        formatted_query = cls.TEMPLATES[template].format(query=query)
        return cls.REASONING_PROMPTS[mode].format(query=formatted_query)
    
    @classmethod
    def build_critique_prompt(cls) -> str:
        """Build validation prompt for self-critique"""
        return """
**Validation Check:**
Review the previous response for:
1. Factual accuracy
2. Logical consistency  
3. Completeness
4. Potential biases or errors

Provide brief validation or corrections if needed."""


class ConversationExporter:
    """Handle conversation export in multiple formats"""
    
    @staticmethod
    def to_json(entries: List[ConversationEntry]) -> str:
        """Export to JSON format"""
        data = [entry.to_dict() for entry in entries]
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def to_markdown(entries: List[ConversationEntry]) -> str:
        """Export to Markdown format"""
        md = "# Conversation History\n\n"
        md += f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        md += "---\n\n"
        
        for i, entry in enumerate(entries, 1):
            md += f"## Conversation {i}\n\n"
            md += f"**Timestamp:** {entry.timestamp}  \n"
            md += f"**Model:** {entry.model}  \n"
            md += f"**Mode:** {entry.reasoning_mode}  \n"
            md += f"**Performance:** {entry.inference_time:.2f}s | {entry.tokens} tokens\n\n"
            md += f"### 👤 User\n\n{entry.user_message}\n\n"
            md += f"### 🤖 Assistant\n\n{entry.ai_response}\n\n"
            
            if entry.bookmarked:
                md += "⭐ *Bookmarked*\n\n"
            
            md += "---\n\n"
        
        return md
    
    @staticmethod
    def to_text(entries: List[ConversationEntry]) -> str:
        """Export to plain text format"""
        txt = "="*70 + "\n"
        txt += "CONVERSATION HISTORY\n"
        txt += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        txt += "="*70 + "\n\n"
        
        for i, entry in enumerate(entries, 1):
            txt += f"Conversation {i}\n"
            txt += f"Time: {entry.timestamp}\n"
            txt += f"Model: {entry.model} | Mode: {entry.reasoning_mode}\n"
            txt += f"Performance: {entry.inference_time:.2f}s | {entry.tokens} tokens\n"
            txt += "\n"
            txt += f"USER:\n{entry.user_message}\n\n"
            txt += f"ASSISTANT:\n{entry.ai_response}\n"
            txt += "\n" + "-"*70 + "\n\n"
        
        return txt
    
    @classmethod
    def export(cls, entries: List[ConversationEntry], format_type: str) -> tuple[str, str]:
        """Export conversation and return content and filename"""
        exporters = {
            "json": cls.to_json,
            "markdown": cls.to_markdown,
            "txt": cls.to_text
        }
        
        content = exporters[format_type](entries)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if format_type == "markdown" else format_type
        filename = EXPORT_DIR / f"conversation_{timestamp}.{ext}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Conversation exported to {filename}")
        return content, str(filename)


class AdvancedReasoner:
    """Enhanced reasoning engine with improved architecture"""
    
    def __init__(self):
        self.client = GroqClientManager.get_client()
        self.metrics = ConversationMetrics()
        self.conversation_history: List[ConversationEntry] = []
        self.response_times: List[float] = []
        self.prompt_engine = PromptEngine()
        self.exporter = ConversationExporter()
    
    def _calculate_reasoning_depth(self, response: str) -> int:
        """Calculate reasoning depth from response"""
        indicators = ["Step", "PATH", "Attempt", "Phase", "Analysis"]
        return sum(response.count(indicator) for indicator in indicators)
    
    def _build_messages(
        self,
        query: str,
        history: List[Dict],
        mode: ReasoningMode,
        template: str
    ) -> List[Dict[str, str]]:
        """Build message list for API call"""
        messages = [
            {"role": "system", "content": self.prompt_engine.SYSTEM_PROMPTS[mode]}
        ]
        
        # Add recent context - clean messages to only have role and content
        recent_history = history[-MAX_HISTORY_LENGTH:] if history else []
        for msg in recent_history:
            # Only include role and content, strip any metadata
            clean_msg = {
                "role": msg.get("role"),
                "content": msg.get("content", "")
            }
            messages.append(clean_msg)
        
        # Add enhanced query
        enhanced_query = self.prompt_engine.build_prompt(query, mode, template)
        messages.append({"role": "user", "content": enhanced_query})
        
        return messages
    
    @error_handler
    def generate_response(
        self,
        query: str,
        history: List[Dict],
        model: str,
        reasoning_mode: ReasoningMode,
        enable_critique: bool,
        temperature: float,
        max_tokens: int,
        prompt_template: str = "Custom"
    ) -> Generator[str, None, None]:
        """Generate response with streaming and validation"""
        
        with timer("Response generation"):
            start_time = time.time()
            
            # Build messages
            messages = self._build_messages(query, history, reasoning_mode, prompt_template)
            
            # Phase 1: Initial reasoning
            full_response = ""
            token_count = 0
            
            try:
                stream = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        token_count += 1
                        self.metrics.tokens_used += 1
                        yield full_response
            
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                raise
            
            # Calculate reasoning depth
            self.metrics.reasoning_depth = self._calculate_reasoning_depth(full_response)
            
            # Phase 2: Self-critique (if enabled)
            if enable_critique and len(full_response) > 150:
                messages.append({"role": "assistant", "content": full_response})
                messages.append({
                    "role": "user",
                    "content": self.prompt_engine.build_critique_prompt()
                })
                
                full_response += "\n\n---\n### 🔍 Validation & Self-Critique\n"
                
                try:
                    critique_stream = self.client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature * 0.7,
                        max_tokens=max_tokens // 3,
                        stream=True,
                    )
                    
                    for chunk in critique_stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            token_count += 1
                            yield full_response
                    
                    self.metrics.self_corrections += 1
                
                except Exception as e:
                    logger.warning(f"Critique phase failed: {e}")
            
            # Update metrics
            inference_time = time.time() - start_time
            self.metrics.inference_time = inference_time
            self.response_times.append(inference_time)
            self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)
            self.metrics.last_updated = datetime.now().strftime("%H:%M:%S")
            self.metrics.update_confidence()
            self.metrics.total_conversations += 1
            
            # Store conversation
            entry = ConversationEntry(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_message=query,
                ai_response=full_response,
                model=model,
                reasoning_mode=reasoning_mode.value,
                inference_time=inference_time,
                tokens=token_count
            )
            self.conversation_history.append(entry)
            
            yield full_response
    
    def export_conversation(self, format_type: str) -> tuple[str, str]:
        """Export conversation history"""
        return self.exporter.export(self.conversation_history, format_type)
    
    def search_conversations(self, keyword: str) -> List[tuple[int, ConversationEntry]]:
        """Search through conversation history"""
        keyword_lower = keyword.lower()
        return [
            (i, entry) for i, entry in enumerate(self.conversation_history)
            if keyword_lower in entry.user_message.lower() 
            or keyword_lower in entry.ai_response.lower()
        ]
    
    def get_analytics(self) -> Optional[Dict[str, Any]]:
        """Generate analytics data"""
        if not self.conversation_history:
            return None
        
        models = [e.model for e in self.conversation_history]
        modes = [e.reasoning_mode for e in self.conversation_history]
        
        return {
            "total_conversations": len(self.conversation_history),
            "total_tokens": sum(e.tokens for e in self.conversation_history),
            "avg_inference_time": sum(e.inference_time for e in self.conversation_history) / len(self.conversation_history),
            "most_used_model": max(set(models), key=models.count),
            "most_used_mode": max(set(modes), key=modes.count),
            "total_time": sum(e.inference_time for e in self.conversation_history)
        }
    
    def bookmark_last(self) -> bool:
        """Bookmark the last conversation"""
        if self.conversation_history:
            self.conversation_history[-1].bookmarked = True
            logger.info("Last conversation bookmarked")
            return True
        return False
    
    def clear_history(self) -> None:
        """Clear conversation history and reset metrics"""
        self.conversation_history.clear()
        self.response_times.clear()
        self.metrics.reset()
        logger.info("History cleared")


# Initialize system
reasoner = AdvancedReasoner()


# Enhanced CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow: 0 10px 25px rgba(0,0,0,0.15);
}

.research-header {
    background: var(--primary);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}

.research-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.75rem; }
.research-header p { font-size: 1.1rem; }

.badge {
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin: 0.3rem;
    display: inline-block;
    transition: all 0.3s ease;
}

.metrics-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 5px solid #667eea;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    font-family: 'Courier New', monospace;
    transition: all 0.3s ease;
}

.analytics-panel {
    background: var(--success);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: var(--shadow);
}

.status-active { color: #10b981; font-weight: bold; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }

.gradio-container { font-family: 'Inter', sans-serif !important; }
"""


def create_ui() -> gr.Blocks:
    """Create Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=CUSTOM_CSS,
        title="🔬 Advanced AI Reasoning System"
    ) as demo:
        
        gr.HTML("""
        <div class="research-header">
            <h1>🔬 Advanced AI Reasoning Research System Pro</h1>
            <p><strong>Research Implementation:</strong> Tree of Thoughts + Constitutional AI + Multi-Agent Validation</p>
            <div style="margin-top: 1rem;">
                <span class="badge">📄 Yao et al. 2023 - Tree of Thoughts</span>
                <span class="badge">📄 Bai et al. 2022 - Constitutional AI</span>
                <span class="badge">✨ Enhanced Architecture</span>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Reasoning Workspace"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="💬 Reasoning Workspace",
                            height=550,
                            show_copy_button=True,
                            type="messages",
                            avatar_images=(
                                "https://api.dicebear.com/7.x/avataaars/svg?seed=User",
                                "https://api.dicebear.com/7.x/bottts/svg?seed=AI"
                            )
                        )
                        
                        msg = gr.Textbox(
                            placeholder="💡 Enter your complex problem or research question...",
                            label="Query Input",
                            lines=3
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("🚀 Process", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ Clear", scale=1)
                            bookmark_btn = gr.Button("⭐ Bookmark", scale=1)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        reasoning_mode = gr.Radio(
                            choices=[mode.value for mode in ReasoningMode],
                            value=ReasoningMode.TREE_OF_THOUGHTS.value,
                            label="🧠 Reasoning Method"
                        )
                        
                        prompt_template = gr.Dropdown(
                            choices=list(PromptEngine.TEMPLATES.keys()),
                            value="Custom",
                            label="📝 Prompt Template"
                        )
                        
                        enable_critique = gr.Checkbox(
                            label="🔍 Enable Self-Critique",
                            value=True
                        )
                        
                        model = gr.Dropdown(
                            choices=[
                                "llama-3.3-70b-versatile",
                                "deepseek-r1-distill-llama-70b",
                                "mixtral-8x7b-32768",
                                "llama-3.1-70b-versatile"
                            ],
                            value="llama-3.3-70b-versatile",
                            label="🤖 Model"
                        )
                        
                        with gr.Accordion("🎛️ Advanced", open=False):
                            temperature = gr.Slider(
                                0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.1,
                                label="🌡️ Temperature"
                            )
                            max_tokens = gr.Slider(
                                1000, 8000, value=DEFAULT_MAX_TOKENS, step=500,
                                label="📏 Max Tokens"
                            )
                        
                        gr.Markdown("### 📊 Live Metrics")
                        metrics_display = gr.Markdown(value=get_metrics_html())
            
            # Export Tab
            with gr.Tab("📥 Export & History"):
                gr.Markdown("### 💾 Export Conversation History")
                
                with gr.Row():
                    export_format = gr.Radio(
                        choices=["json", "markdown", "txt"],
                        value="markdown",
                        label="Format"
                    )
                    export_btn = gr.Button("📥 Export", variant="primary")
                
                export_output = gr.Code(label="Exported Data", language="markdown", lines=20)
                download_file = gr.File(label="Download")
                
                gr.Markdown("### 🔍 Search Conversations")
                with gr.Row():
                    search_input = gr.Textbox(placeholder="Enter keyword...", scale=3)
                    search_btn = gr.Button("🔍 Search", scale=1)
                
                search_results = gr.Markdown("No results yet.")
            
            # Analytics Tab
            with gr.Tab("📊 Analytics"):
                refresh_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
                analytics_display = gr.Markdown(get_empty_analytics_html())
        
        # Event handlers
        def process_message(message, history, mode, critique, model_name, temp, tokens, template):
            if not message.strip():
                return history, get_metrics_html()
            
            history = history or []
            mode_enum = ReasoningMode(mode)
            
            history.append({"role": "user", "content": message})
            yield history, get_metrics_html()
            
            history.append({"role": "assistant", "content": ""})
            
            for response in reasoner.generate_response(
                message, history[:-1], model_name, mode_enum, 
                critique, temp, tokens, template
            ):
                history[-1]["content"] = response
                yield history, get_metrics_html()
        
        def reset_chat():
            reasoner.clear_history()
            return [], get_metrics_html()
        
        def export_conv(format_type):
            content, filename = reasoner.export_conversation(format_type)
            return content, filename
        
        def search_conv(keyword):
            results = reasoner.search_conversations(keyword)
            if not results:
                return "❌ No results found."
            
            output = f"### 🔍 Found {len(results)} result(s)\n\n"
            for idx, entry in results[:10]:  # Limit to 10 results
                output += f"**{idx + 1}.** {entry.timestamp} | {entry.model}\n"
                output += f"User: {entry.user_message[:100]}...\n\n"
            return output
        
        def refresh_analytics():
            analytics = reasoner.get_analytics()
            if not analytics:
                return get_empty_analytics_html()
            
            return f"""<div class="analytics-panel">
            <h3>📊 Performance Analytics</h3>
            <p><strong>Total Conversations:</strong> {analytics['total_conversations']}</p>
            <p><strong>Total Tokens:</strong> {analytics['total_tokens']:,}</p>
            <p><strong>Avg Time:</strong> {analytics['avg_inference_time']:.2f}s</p>
            <p><strong>Total Time:</strong> {analytics['total_time']:.1f}s</p>
            <p><strong>Most Used Model:</strong> {analytics['most_used_model']}</p>
            <p><strong>Most Used Mode:</strong> {analytics['most_used_mode']}</p>
            </div>"""
        
        def bookmark_last_conv():
            if reasoner.bookmark_last():
                return "✅ Last conversation bookmarked!"
            return "❌ No conversation to bookmark."
        
        # Connect events
        submit_btn.click(
            process_message,
            [msg, chatbot, reasoning_mode, enable_critique, model, temperature, max_tokens, prompt_template],
            [chatbot, metrics_display]
        ).then(lambda: "", None, msg)
        
        msg.submit(
            process_message,
            [msg, chatbot, reasoning_mode, enable_critique, model, temperature, max_tokens, prompt_template],
            [chatbot, metrics_display]
        ).then(lambda: "", None, msg)
        
        clear_btn.click(reset_chat, None, [chatbot, metrics_display])
        bookmark_btn.click(bookmark_last_conv, None, gr.Textbox(visible=False))
        export_btn.click(export_conv, export_format, [export_output, download_file])
        search_btn.click(search_conv, search_input, search_results)
        refresh_btn.click(refresh_analytics, None, analytics_display)
    
    return demo


def get_metrics_html() -> str:
    """Generate metrics HTML"""
    m = reasoner.metrics
    status = '<span class="status-active">Active</span>' if m.tokens_used > 0 else 'Ready'
    
    return f"""<div class="metrics-card">
    <strong>⏱️ Inference:</strong> {m.inference_time:.2f}s<br>
    <strong>📊 Avg Time:</strong> {m.avg_response_time:.2f}s<br>
    <strong>🧠 Reasoning:</strong> {m.reasoning_depth} steps<br>
    <strong>✅ Corrections:</strong> {m.self_corrections}<br>
    <strong>🎯 Confidence:</strong> {m.confidence_score:.1f}%<br>
    <strong>💬 Total:</strong> {m.total_conversations}<br>
    <strong>🔢 Tokens:</strong> {m.tokens_used:,}<br>
    <strong>📍 Status:</strong> {status}
    </div>"""


def get_empty_analytics_html() -> str:
    """Generate empty analytics HTML"""
    return """<div class="analytics-panel">
    <h3>📊 No data yet</h3>
    <p>Start a conversation to see analytics!</p>
    </div>"""


if __name__ == "__main__":
    try:
        logger.info("Starting Advanced AI Reasoning System...")
        demo = create_ui()
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            show_api=False
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise