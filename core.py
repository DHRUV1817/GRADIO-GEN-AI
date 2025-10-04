import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Generator, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from functools import wraps, lru_cache
from contextlib import contextmanager
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from groq import Groq

from config import logger, AppConfig, ReasoningMode, ModelConfig

# Thread-safe caching decorator
class ResponseCache:
    """Thread-safe LRU cache for API responses"""
    def __init__(self, maxsize: int = 100, ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    logger.debug(f"Cache hit for key: {key[:20]}...")
                    return value
                else:
                    del self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with timestamp"""
        with self.lock:
            if len(self.cache) >= self.maxsize:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())
            logger.debug(f"Cached response for key: {key[:20]}...")
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 2),
                "size": len(self.cache)
            }

class RateLimiter:
    """Token bucket rate limiter"""
    def __init__(self, max_requests: int = 50, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> Tuple[bool, Optional[float]]:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.window:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True, None
            else:
                wait_time = self.window - (now - self.requests[0])
                return False, wait_time
    
    def reset(self) -> None:
        """Reset rate limiter"""
        with self.lock:
            self.requests.clear()

@dataclass
class ConversationMetrics:
    """Enhanced metrics with advanced tracking"""
    reasoning_depth: int = 0
    self_corrections: int = 0
    confidence_score: float = 0.0
    inference_time: float = 0.0
    tokens_used: int = 0
    tokens_per_second: float = 0.0
    reasoning_paths_explored: int = 0
    total_conversations: int = 0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    retry_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    session_start: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model_switches: int = 0
    mode_switches: int = 0
    peak_tokens: int = 0
    total_latency: float = 0.0
    
    def update_confidence(self) -> None:
        """Calculate confidence based on multiple factors"""
        depth_score = min(30, self.reasoning_depth * 5)
        correction_score = min(20, self.self_corrections * 10)
        speed_score = min(25, 25 / max(1, self.avg_response_time))
        consistency_score = 25
        self.confidence_score = min(95.0, depth_score + correction_score + speed_score + consistency_score)
    
    def update_tokens_per_second(self, tokens: int, time_taken: float) -> None:
        """Calculate tokens per second"""
        if time_taken > 0:
            self.tokens_per_second = tokens / time_taken
    
    def reset(self) -> None:
        """Reset metrics for new session"""
        self.__init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ConversationEntry:
    """Enhanced conversation entry with metadata"""
    timestamp: str
    user_message: str
    ai_response: str
    model: str
    reasoning_mode: str
    inference_time: float
    tokens: int
    feedback: str = ""
    tags: List[str] = field(default_factory=list)
    rating: Optional[int] = None
    session_id: str = ""
    conversation_id: str = ""
    parent_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    cache_hit: bool = False
    error_occurred: bool = False
    retry_count: int = 0
    tokens_per_second: float = 0.0
    
    def __post_init__(self):
        """Generate unique IDs"""
        if not self.conversation_id:
            self.conversation_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique conversation ID"""
        content = f"{self.timestamp}{self.user_message}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sanitization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create instance from dictionary"""
        return cls(**data)
    
    def add_tag(self, tag: str) -> None:
        """Add tag to conversation"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_rating(self, rating: int) -> None:
        """Set user rating (1-5)"""
        if 1 <= rating <= 5:
            self.rating = rating

def error_handler(func):
    """Enhanced error handling decorator with retries"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = AppConfig.MAX_RETRIES
        retry_delay = AppConfig.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__} (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    error_msg = f"⚠️ **System Error:** {str(e)}\n\n"
                    
                    if "api" in str(e).lower() or "key" in str(e).lower():
                        error_msg += "Please verify your GROQ_API_KEY in the .env file."
                    elif "rate" in str(e).lower() or "limit" in str(e).lower():
                        error_msg += "Rate limit exceeded. Please wait a moment and try again."
                    elif "timeout" in str(e).lower():
                        error_msg += "Request timed out. Please try again."
                    else:
                        error_msg += "Please try again or contact support if the issue persists."
                    
                    return error_msg
    return wrapper

@contextmanager
def timer(operation: str = "Operation"):
    """Enhanced context manager for timing operations"""
    start = time.time()
    logger.info(f"Starting: {operation}")
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"Completed: {operation} in {duration:.3f}s")

def validate_input(text: str, max_length: int = 10000) -> Tuple[bool, Optional[str]]:
    """Validate user input"""
    if not text or not text.strip():
        return False, "Input cannot be empty"
    
    if len(text) > max_length:
        return False, f"Input too long (max {max_length} characters)"
    
    suspicious_patterns = ["<script", "javascript:", "onerror=", "onclick="]
    text_lower = text.lower()
    for pattern in suspicious_patterns:
        if pattern in text_lower:
            return False, "Input contains potentially unsafe content"
    
    return True, None

class GroqClientManager:
    """Enhanced singleton manager for Groq client"""
    _instance: Optional[Groq] = None
    _lock = threading.Lock()
    _initialized = False
    _health_check_time: Optional[float] = None
    _health_check_interval = 300
    
    @classmethod
    def get_client(cls) -> Groq:
        """Get or create Groq client instance with health check"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._initialize_client()
        
        if cls._should_health_check():
            cls._perform_health_check()
        
        return cls._instance
    
    @classmethod
    def _initialize_client(cls) -> None:
        """Initialize Groq client"""
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            logger.error("GROQ_API_KEY not found in environment")
            raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
        
        try:
            cls._instance = Groq(api_key=api_key, timeout=AppConfig.REQUEST_TIMEOUT)
            cls._initialized = True
            cls._health_check_time = time.time()
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    @classmethod
    def _should_health_check(cls) -> bool:
        """Check if health check is needed"""
        if not cls._health_check_time:
            return True
        return time.time() - cls._health_check_time > cls._health_check_interval
    
    @classmethod
    def _perform_health_check(cls) -> None:
        """Perform health check on client"""
        try:
            if cls._instance:
                cls._health_check_time = time.time()
                logger.debug("Health check passed")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            cls._instance = None
            cls._initialized = False

class PromptEngine:
    """Enhanced centralized prompt management"""
    
    SYSTEM_PROMPTS = {
        ReasoningMode.TREE_OF_THOUGHTS: """You are an advanced reasoning system using Tree of Thoughts methodology.
Explore multiple reasoning paths systematically before converging on the best solution.
Always show your thought process explicitly.""",
        
        ReasoningMode.CHAIN_OF_THOUGHT: """You are a systematic problem solver using Chain of Thought reasoning.
Break down complex problems into clear, logical steps with explicit reasoning.""",
        
        ReasoningMode.SELF_CONSISTENCY: """You are a consistency-focused reasoning system.
Generate multiple independent solutions and identify the most consistent answer.""",
        
        ReasoningMode.REFLEXION: """You are a self-reflective AI system.
Solve problems, critique your own reasoning, and refine your solutions iteratively.""",
        
        ReasoningMode.DEBATE: """You are a multi-agent debate system.
Present multiple perspectives and synthesize the strongest arguments.""",
        
        ReasoningMode.ANALOGICAL: """You are an analogical reasoning system.
Find similar problems and apply their solutions."""
    }
    
    TEMPLATES = {
        "Code Review": {
            "prompt": "Analyze the following code for bugs, performance issues, and best practices:\n\n{query}",
            "context": "code_analysis"
        },
        "Research Summary": {
            "prompt": "Provide a comprehensive research summary on:\n\n{query}\n\nInclude key findings, methodologies, and implications.",
            "context": "research"
        },
        "Problem Solving": {
            "prompt": "Solve this problem step-by-step with detailed explanations:\n\n{query}",
            "context": "problem_solving"
        },
        "Creative Writing": {
            "prompt": "Generate creative content based on:\n\n{query}\n\nBe imaginative and engaging.",
            "context": "creative"
        },
        "Data Analysis": {
            "prompt": "Analyze this data/scenario and provide insights:\n\n{query}",
            "context": "analysis"
        },
        "Debugging": {
            "prompt": "Debug this code/issue systematically:\n\n{query}",
            "context": "debugging"
        },
        "Custom": {
            "prompt": "{query}",
            "context": "general"
        }
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

**Refined Solution:** [Improved answer based on reflection]""",

        ReasoningMode.DEBATE: """
💬 **Multi-Agent Debate**

Problem: {query}

**Position A:** [Strongest case for one approach]
**Position B:** [Critical examination]
**Synthesis:** [Balanced conclusion]""",

        ReasoningMode.ANALOGICAL: """
🔄 **Analogical Reasoning**

Problem: {query}

**Similar Problems:** [Identify analogous situations]
**Solution Transfer:** [Adapt known solutions]
**Final Answer:** [Solution derived from analogy]"""
    }
    
    @classmethod
    def build_prompt(cls, query: str, mode: ReasoningMode, template: str) -> str:
        """Build enhanced reasoning prompt"""
        template_data = cls.TEMPLATES.get(template, cls.TEMPLATES["Custom"])
        formatted_query = template_data["prompt"].format(query=query)
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
    
    @classmethod
    def get_template_context(cls, template: str) -> str:
        """Get context for template"""
        return cls.TEMPLATES.get(template, {}).get("context", "general")

class ConversationExporter:
    """Enhanced conversation export with multiple formats including PDF"""
    
    @staticmethod
    def to_json(entries: List[ConversationEntry], pretty: bool = True) -> str:
        """Export to JSON format"""
        data = [entry.to_dict() for entry in entries]
        indent = 2 if pretty else None
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def to_markdown(entries: List[ConversationEntry], include_metadata: bool = True) -> str:
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
    
    @staticmethod
    def to_pdf(entries: List[ConversationEntry], filename: str) -> str:
        """Export to PDF format"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            from reportlab.lib.colors import HexColor
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=HexColor('#667eea'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=HexColor('#764ba2'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            user_style = ParagraphStyle(
                'UserStyle',
                parent=styles['Normal'],
                fontSize=11,
                textColor=HexColor('#2c3e50'),
                leftIndent=20,
                spaceAfter=10
            )
            
            ai_style = ParagraphStyle(
                'AIStyle',
                parent=styles['Normal'],
                fontSize=11,
                textColor=HexColor('#34495e'),
                leftIndent=20,
                spaceAfter=10
            )
            
            meta_style = ParagraphStyle(
                'MetaStyle',
                parent=styles['Normal'],
                fontSize=9,
                textColor=HexColor('#7f8c8d'),
                spaceAfter=6
            )
            
            story.append(Paragraph("🔬 AI Reasoning Chat History", title_style))
            story.append(Paragraph(
                f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                meta_style
            ))
            story.append(Spacer(1, 0.3*inch))
            
            for i, entry in enumerate(entries, 1):
                story.append(Paragraph(f"Conversation {i}", heading_style))
                
                meta_text = f"<b>Time:</b> {entry.timestamp} | <b>Model:</b> {entry.model} | <b>Mode:</b> {entry.reasoning_mode}"
                story.append(Paragraph(meta_text, meta_style))
                
                perf_text = f"<b>Performance:</b> {entry.inference_time:.2f}s | {entry.tokens} tokens | {entry.tokens_per_second:.1f} tok/s"
                story.append(Paragraph(perf_text, meta_style))
                story.append(Spacer(1, 0.1*inch))
                
                story.append(Paragraph("👤 <b>User:</b>", user_style))
                user_msg = entry.user_message.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
                if len(user_msg) > 3000:
                    user_msg = user_msg[:3000] + "... (truncated)"
                story.append(Paragraph(user_msg, user_style))
                story.append(Spacer(1, 0.15*inch))
                
                story.append(Paragraph("🤖 <b>Assistant:</b>", ai_style))
                ai_resp = entry.ai_response.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
                if len(ai_resp) > 5000:
                    ai_resp = ai_resp[:5000] + "... (truncated)"
                story.append(Paragraph(ai_resp, ai_style))
                
                if i < len(entries):
                    story.append(PageBreak())
            
            doc.build(story)
            logger.info(f"PDF exported to {filename}")
            return filename
            
        except ImportError:
            error_msg = "reportlab library not installed. Run: pip install reportlab"
            logger.error(error_msg)
            return ""
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            return ""
    
    @classmethod
    def export(cls, entries: List[ConversationEntry], format_type: str, 
               include_metadata: bool = True) -> Tuple[str, str]:
        """Export conversation and return content and filename"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "pdf":
            ext = "pdf"
            filename = AppConfig.EXPORT_DIR / f"conversation_{timestamp}.{ext}"
            result = cls.to_pdf(entries, str(filename))
            if result:
                return "PDF exported successfully! Check the exports folder.", str(filename)
            else:
                return "PDF export failed. Install reportlab: pip install reportlab", ""
        
        exporters = {
            "json": lambda: cls.to_json(entries),
            "markdown": lambda: cls.to_markdown(entries, include_metadata),
            "txt": lambda: cls.to_text(entries)
        }
        
        if format_type not in exporters:
            format_type = "markdown"
        
        content = exporters[format_type]()
        ext = "md" if format_type == "markdown" else format_type
        filename = AppConfig.EXPORT_DIR / f"conversation_{timestamp}.{ext}"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Conversation exported to {filename}")
            return content, str(filename)
        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            return f"Error: {str(e)}", ""
    
    @staticmethod
    def create_backup(entries: List[ConversationEntry]) -> str:
        """Create automatic backup"""
        if not entries:
            return ""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = AppConfig.BACKUP_DIR / f"backup_{timestamp}.json"
            
            data = [entry.to_dict() for entry in entries]
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Backup created: {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return ""

class AdvancedReasoner:
    """Enhanced reasoning engine with caching, rate limiting, and advanced features"""
    
    def __init__(self):
        self.client = GroqClientManager.get_client()
        self.metrics = ConversationMetrics()
        self.conversation_history: List[ConversationEntry] = []
        self.response_times: List[float] = []
        self.prompt_engine = PromptEngine()
        self.exporter = ConversationExporter()
        
        self.cache = ResponseCache(maxsize=AppConfig.CACHE_SIZE, ttl=AppConfig.CACHE_TTL)
        self.rate_limiter = RateLimiter(
            max_requests=AppConfig.RATE_LIMIT_REQUESTS,
            window=AppConfig.RATE_LIMIT_WINDOW
        )
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.mode_usage: Dict[str, int] = defaultdict(int)
        self.error_log: List[Dict[str, Any]] = []
        
        logger.info(f"AdvancedReasoner initialized with session ID: {self.session_id}")
    
    def _generate_cache_key(self, query: str, model: str, mode: str, 
                           temp: float, template: str) -> str:
        """Generate cache key for request"""
        content = f"{query}|{model}|{mode}|{temp:.2f}|{template}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _calculate_reasoning_depth(self, response: str) -> int:
        """Calculate reasoning depth from response"""
        indicators = {
            "Step": 3, "PATH": 4, "Attempt": 3, "Phase": 3,
            "Analysis": 2, "Consider": 1, "Therefore": 2,
            "Conclusion": 2, "Evidence": 2, "Reasoning": 1
        }
        
        depth = 0
        for indicator, weight in indicators.items():
            depth += response.count(indicator) * weight
        
        return min(depth, 100)
    
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
        
        recent_history = history[-AppConfig.MAX_HISTORY_LENGTH:] if history else []
        for msg in recent_history:
            clean_msg = {
                "role": msg.get("role"),
                "content": msg.get("content", "")
            }
            messages.append(clean_msg)
        
        enhanced_query = self.prompt_engine.build_prompt(query, mode, template)
        messages.append({"role": "user", "content": enhanced_query})
        
        return messages
    
    def _log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with context"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        }
        self.error_log.append(error_entry)
        self.metrics.error_count += 1
        logger.error(f"Error logged: {error_entry}")
    
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
        prompt_template: str = "Custom",
        use_cache: bool = True
    ) -> Generator[str, None, None]:
        """Generate response with advanced features"""
        
        is_valid, error_msg = validate_input(query)
        if not is_valid:
            yield f"❌ **Validation Error:** {error_msg}"
            return
        
        allowed, wait_time = self.rate_limiter.is_allowed()
        if not allowed:
            yield f"⏸️ **Rate Limit:** Please wait {wait_time:.1f} seconds."
            return
        
        cache_key = self._generate_cache_key(query, model, reasoning_mode.value, temperature, prompt_template)
        if use_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.metrics.cache_hits += 1
                logger.info("Returning cached response")
                yield cached_response
                return
        
        self.metrics.cache_misses += 1
        
        with timer(f"Response generation for {model}"):
            start_time = time.time()
            messages = self._build_messages(query, history, reasoning_mode, prompt_template)
            
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
                self._log_error(e, {
                    "query": query[:100],
                    "model": model,
                    "mode": reasoning_mode.value
                })
                raise
            
            inference_time = time.time() - start_time
            self.metrics.reasoning_depth = self._calculate_reasoning_depth(full_response)
            self.metrics.update_tokens_per_second(token_count, inference_time)
            self.metrics.peak_tokens = max(self.metrics.peak_tokens, token_count)
            
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
            
            final_inference_time = time.time() - start_time
            self.metrics.inference_time = final_inference_time
            self.metrics.total_latency += final_inference_time
            self.response_times.append(final_inference_time)
            self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)
            self.metrics.last_updated = datetime.now().strftime("%H:%M:%S")
            self.metrics.update_confidence()
            self.metrics.total_conversations += 1
            
            self.model_usage[model] += 1
            self.mode_usage[reasoning_mode.value] += 1
            
            tokens_per_sec = token_count / final_inference_time if final_inference_time > 0 else 0
            entry = ConversationEntry(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_message=query,
                ai_response=full_response,
                model=model,
                reasoning_mode=reasoning_mode.value,
                inference_time=final_inference_time,
                tokens=token_count,
                session_id=self.session_id,
                temperature=temperature,
                max_tokens=max_tokens,
                cache_hit=False,
                tokens_per_second=tokens_per_sec
            )
            
            self.conversation_history.append(entry)
            
            if use_cache:
                self.cache.set(cache_key, full_response)
            
            if len(self.conversation_history) % 10 == 0:
                try:
                    self.exporter.create_backup(self.conversation_history)
                except Exception as e:
                    logger.warning(f"Auto-backup failed: {e}")
            
            if len(self.conversation_history) > AppConfig.MAX_CONVERSATION_STORAGE:
                self.conversation_history = self.conversation_history[-AppConfig.MAX_CONVERSATION_STORAGE:]
                logger.info(f"Trimmed history to {AppConfig.MAX_CONVERSATION_STORAGE} entries")
            
            yield full_response
    
    def export_conversation(self, format_type: str, include_metadata: bool = True) -> Tuple[str, str]:
        """Export conversation history"""
        if not self.conversation_history:
            return "No conversations to export.", ""
        
        try:
            return self.exporter.export(self.conversation_history, format_type, include_metadata)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return f"Export failed: {str(e)}", ""
    
    def export_current_chat_pdf(self) -> Optional[str]:
        """Export current chat as PDF - for quick download button"""
        if not self.conversation_history:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = AppConfig.EXPORT_DIR / f"chat_{timestamp}.pdf"
        
        result = self.exporter.to_pdf(self.conversation_history, str(filename))
        return result if result else None
    
    def search_conversations(self, keyword: str) -> List[Tuple[int, ConversationEntry]]:
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
        total_time = sum(e.inference_time for e in self.conversation_history)
        total_tokens = sum(e.tokens for e in self.conversation_history)
        
        return {
            "session_id": self.session_id,
            "total_conversations": len(self.conversation_history),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "avg_inference_time": self.metrics.avg_response_time,
            "peak_tokens": self.metrics.peak_tokens,
            "most_used_model": max(set(models), key=models.count),
            "most_used_mode": max(set(modes), key=modes.count),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "error_count": self.metrics.error_count
        }
    
    def clear_history(self) -> None:
        """Clear conversation history and reset metrics"""
        if self.conversation_history:
            try:
                self.exporter.create_backup(self.conversation_history)
            except Exception as e:
                logger.warning(f"Failed to backup before clearing: {e}")
        
        self.conversation_history.clear()
        self.response_times.clear()
        self.metrics.reset()
        self.cache.clear()
        self.rate_limiter.reset()
        self.model_usage.clear()
        self.mode_usage.clear()
        
        logger.info("History cleared and metrics reset")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.executor.shutdown(wait=False)
        logger.info("AdvancedReasoner cleanup completed")
