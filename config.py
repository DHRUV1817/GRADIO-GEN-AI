import logging
from pathlib import Path
from enum import Enum
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup advanced logging with rotation"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    file_handler = RotatingFileHandler(
        'reasoning_system.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

class AppConfig:
    """Centralized application configuration"""
    MAX_HISTORY_LENGTH: int = 10
    MAX_CONVERSATION_STORAGE: int = 1000
    DEFAULT_TEMPERATURE: float = 0.7
    MIN_TEMPERATURE: float = 0.0
    MAX_TEMPERATURE: float = 2.0
    DEFAULT_MAX_TOKENS: int = 4000
    MIN_TOKENS: int = 100
    MAX_TOKENS: int = 32000
    REQUEST_TIMEOUT: int = 60
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    CACHE_SIZE: int = 100
    CACHE_TTL: int = 3600
    RATE_LIMIT_REQUESTS: int = 50
    RATE_LIMIT_WINDOW: int = 60
    EXPORT_DIR: Path = Path("exports")
    BACKUP_DIR: Path = Path("backups")
    MAX_EXPORT_SIZE_MB: int = 50
    THEME_PRIMARY: str = "purple"
    THEME_SECONDARY: str = "blue"
    AUTO_SAVE_INTERVAL: int = 300
    ENABLE_ANALYTICS: bool = True
    ANALYTICS_BATCH_SIZE: int = 10
    
    @classmethod
    def validate(cls) -> bool:
        try:
            assert cls.MIN_TEMPERATURE <= cls.DEFAULT_TEMPERATURE <= cls.MAX_TEMPERATURE
            assert cls.MIN_TOKENS <= cls.DEFAULT_MAX_TOKENS <= cls.MAX_TOKENS
            assert cls.MAX_HISTORY_LENGTH > 0
            return True
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def create_directories(cls) -> None:
        cls.EXPORT_DIR.mkdir(exist_ok=True)
        cls.BACKUP_DIR.mkdir(exist_ok=True)
        logger.info("Application directories initialized")

AppConfig.create_directories()
AppConfig.validate()

class ReasoningMode(Enum):
    """Research-aligned reasoning methodologies"""
    TREE_OF_THOUGHTS = "Tree of Thoughts (ToT)"
    CHAIN_OF_THOUGHT = "Chain of Thought (CoT)"
    SELF_CONSISTENCY = "Self-Consistency Sampling"
    REFLEXION = "Reflexion + Self-Correction"
    DEBATE = "Multi-Agent Debate"
    ANALOGICAL = "Analogical Reasoning"

class ModelConfig(Enum):
    """Available models with specifications"""
    # Original Models
    LLAMA_70B = ("llama-3.3-70b-versatile", 70, 8000, "Best overall")
    DEEPSEEK_70B = ("deepseek-r1-distill-llama-70b", 70, 8000, "Optimized reasoning")
    MIXTRAL_8X7B = ("mixtral-8x7b-32768", 47, 32768, "Long context")
    LLAMA_70B_V31 = ("llama-3.1-70b-versatile", 70, 8000, "Stable")
    GEMMA_9B = ("gemma2-9b-it", 9, 8192, "Fast")
    
    # Meta / Llama
    LLAMA_3_1_8B_INSTANT = ("llama-3.1-8b-instant", 8, 131072, "Fast responses")
    LLAMA_4_MAVERICK_17B = ("meta-llama/llama-4-maverick-17b-128k", 17, 131072, "Llama 4 experimental")
    LLAMA_4_SCOUT_17B = ("meta-llama/llama-4-scout-17b-16e-instruct", 17, 16384, "Llama 4 scout model")
    LLAMA_GUARD_4_12B = ("meta-llama/llama-guard-4-12b", 12, 8192, "Safety/Guard model")
    LLAMA_PROMPT_GUARD_2_22M = ("meta-llama/llama-prompt-guard-2-22m", 0, 8192, "Prompt safety (22M)")
    LLAMA_PROMPT_GUARD_2_86M = ("meta-llama/llama-prompt-guard-2-86m", 0, 8192, "Prompt safety (86M)")
    
    # Moonshot AI
    KIMI_K2_INSTRUCT_DEPRECATED = ("moonshotai/kimi-k2-instruct", 0, 200000, "Long context (Deprecated)")
    KIMI_K2_INSTRUCT_0905 = ("moonshotai/kimi-k2-instruct-0905", 0, 200000, "Long context")
    
    # OpenAI
    GPT_OSS_120B = ("openai/gpt-oss-120b", 120, 8192, "OpenAI open source model")
    GPT_OSS_20B = ("openai/gpt-oss-20b", 20, 8192, "OpenAI open source model")
    
    # Qwen
    QWEN3_32B = ("qwen/qwen3-32b", 32, 32768, "Qwen 3 model")
    
    # Groq
    GROQ_COMPOUND = ("groq/compound", 0, 8192, "Groq compound model")
    GROQ_COMPOUND_MINI = ("groq/compound-mini", 0, 8192, "Groq mini compound model")
    
    def __init__(self, model_id: str, params_b: int, max_context: int, description: str):
        self.model_id = model_id
        self.params_b = params_b
        self.max_context = max_context
        self.description = description

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.15);
    --border-radius: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.research-header {
    background: var(--primary-gradient);
    padding: 3rem 2.5rem;
    border-radius: var(--border-radius);
    color: white;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    animation: slideDown 0.6s ease-out;
}

.research-header h1 { 
    font-size: 2.5rem; 
    font-weight: 800; 
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.badge {
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 25px;
    font-size: 0.9rem;
    margin: 0.3rem;
    display: inline-block;
    transition: var(--transition);
    border: 1px solid rgba(255,255,255,0.2);
}

.badge:hover {
    transform: translateY(-2px);
    background: rgba(255,255,255,0.35);
}

.metrics-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-left: 5px solid #667eea;
    padding: 1.8rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
    font-family: 'JetBrains Mono', monospace;
    transition: var(--transition);
    color: #2c3e50 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.metrics-card strong {
    color: #1a1a1a !important;
    font-weight: 600;
}

.metrics-card:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

.analytics-panel {
    background: var(--success-gradient);
    color: white;
    padding: 2rem;
    border-radius: var(--border-radius);
    animation: fadeIn 0.5s ease-out;
    box-shadow: var(--shadow-lg);
}

.analytics-panel h3 {
    color: white !important;
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

.analytics-panel p {
    color: rgba(255,255,255,0.95) !important;
    line-height: 1.6;
}

.analytics-panel strong {
    color: white !important;
    font-weight: 600;
}

.status-active { 
    color: #10b981 !important; 
    font-weight: bold; 
    animation: pulse 2s infinite;
    text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.gradio-container { 
    font-family: 'Inter', sans-serif !important;
    max-width: 1600px !important;
}

.gr-button { 
    transition: var(--transition) !important; 
}

.gr-button:hover { 
    transform: translateY(-2px) !important; 
}

.gr-markdown {
    color: #2c3e50 !important;
}

.gr-markdown strong {
    color: #1a1a1a !important;
}
"""

logger.info("Enhanced configuration initialized")