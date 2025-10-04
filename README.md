🔬 Advanced AI Reasoning Research System
https://opensource.org/licenses/MIT
https://www.python.org/downloads/
https://gradio.app/
https://groq.com
An open-source research platform that implements cutting-edge AI reasoning methodologies including Tree of Thoughts, Constitutional AI, and multi-agent debate patterns. Features a modern web interface, real-time streaming, and comprehensive analytics.
🎯 What This Project Does
Multi-Strategy Reasoning: Apply different reasoning approaches to the same problem
Self-Critique System: AI reviews and improves its own responses
Real-time Analytics: Track reasoning depth, confidence, and performance metrics
Export & Documentation: Save conversations as PDF, Markdown, or JSON
Production Ready: Caching, rate limiting, error handling, and automatic backups
🚀 Quick Start (2 Minutes)
Prerequisites
Python 3.8+
Groq API key (free at console.groq.com)
Installation
bash
Copy
# Clone repository
git clone https://github.com/your-username/ai-reasoning-system.git
cd ai-reasoning-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# Launch system
python main.py
Visit http://localhost:7860 in your browser.
📊 Reasoning Strategies
Table
Copy
Method	Description	Best For
Tree of Thoughts	Explores multiple reasoning paths systematically	Complex problems with multiple solutions
Chain of Thought	Step-by-step transparent reasoning	Mathematical problems, logic puzzles
Self-Consistency	Generates multiple answers and finds consensus	Factual questions, reliability important
Reflexion	Self-critique and iterative improvement	Creative writing, analysis tasks
Multi-Agent Debate	Presents multiple perspectives	Ethical dilemmas, policy questions
Analogical Reasoning	Finds similar problems and adapts solutions	Novel problems, innovation tasks
🎥 Demo Features
Real-time Interface
Streaming Responses: Watch reasoning unfold in real-time
Live Metrics: See inference time, tokens/second, reasoning depth
Interactive Controls: Switch models, adjust temperature, enable critique
Modern Design: Clean, responsive interface with dark theme
Analytics Dashboard
Session performance metrics
Model usage distribution
Cache hit rates
Error tracking and retry statistics
Export Options
PDF: Professional reports with formatting
Markdown: GitHub-friendly documentation
JSON: Machine-readable data
Plain Text: Simple conversation logs
🔧 Configuration
Key settings in config.py:
Python
Copy
MAX_HISTORY_LENGTH = 10          # Messages in context
CACHE_SIZE = 100                 # Cached responses
RATE_LIMIT_REQUESTS = 50         # Per minute
DEFAULT_TEMPERATURE = 0.7        # Creativity level
MAX_TOKENS = 4000                # Response length
🏗️ Architecture
Copy
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  Core Engine    │    │   Groq API      │
│                 │    │                 │    │                 │
│ • Chat Interface│◄──►│ • Reasoning     │◄──►│ • LLM Models    │
│ • Controls      │    │ • Caching       │    │ • Streaming     │
│ • Metrics       │    │ • Rate Limiting │    │ • Token Count   │
│ • Export        │    │ • Error Handling│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
📈 Performance
Cold Start: ~2 seconds
Time to First Token: 0.3-1.2 seconds
Throughput: Up to 100 tokens/second
Memory Usage: ~100MB base + conversation history
Concurrent Users: Limited by Groq rate limits (50 req/min)
🧪 Example Use Cases
Research Analysis
Copy
User: "Analyze the impact of remote work on productivity"
System: Uses Tree of Thoughts to explore economic, psychological, and technological factors
Code Review
Copy
User: "Review this Python function for errors"
System: Applies Chain of Thought to identify bugs, suggest improvements
Creative Writing
Copy
User: "Write a story about AI consciousness"
System: Uses Reflexion to draft, critique, and refine the narrative
Decision Making
Copy
User: "Should we implement a four-day work week?"
System: Multi-Agent Debate presents management and employee perspectives
📚 Research Foundation
Built on seminal papers:
Tree of Thoughts (Yao et al., 2023) - Systematic exploration
Constitutional AI (Bai et al., 2022) - Self-critique mechanisms
Chain of Thought (Wei et al., 2022) - Transparent reasoning
Reflexion (Shinn et al., 2023) - Iterative improvement
Self-Consistency (Wang et al., 2022) - Consensus building
🔍 Project Structure
Copy
ai-reasoning-system/
├── main.py              # Gradio interface and event handlers
├── core.py              # Business logic and reasoning engine
├── config.py            # Configuration and constants
├── requirements.txt     # Dependencies
├── README.md           # This file
├── .env                # API keys (created by user)
├── exports/            # Exported conversations
├── backups/            # Automatic backups
└── reasoning_system.log # Application logs
🧪 Development
Running Tests
bash
Copy
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=core
Adding New Reasoning Mode
Add enum value in ReasoningMode
Add system prompt in PromptEngine.SYSTEM_PROMPTS
Add reasoning template in PromptEngine.REASONING_PROMPTS
Update UI choices in main.py
Custom Models
Add to ModelConfig enum:
Python
Copy
CUSTOM_MODEL = ("custom-model-id", parameters, context_length, "Description")
🔧 Troubleshooting
Table
Copy
Issue	Solution
API Key Error	Check .env file format: GROQ_API_KEY=gsk_...
Rate Limit Hit	Wait 60 seconds or reduce request frequency
Memory Issues	Reduce MAX_CONVERSATION_STORAGE in config
PDF Export Fails	Install reportlab: pip install reportlab
Port Already in Use	Change port: python main.py --port 7861
📄 License
MIT License - see LICENSE file for details.
🎓 Academic Use
Perfect for:
Final year projects
Research demonstrations
AI methodology studies
Human-AI interaction experiments
Citation:
bibtex
Copy
@software{ai_reasoning_system_2025,
  title = {Advanced AI Reasoning Research System},
  year = {2025},
  url = {https://github.com/your-username/ai-reasoning-system}
}
🤝 Contributing
Fork the repository
Create feature branch: git checkout -b feature-name
Commit changes: git commit -m "Add feature"
Push to branch: git push origin feature-name
Submit Pull Request
📞 Support
Create an issue for bugs or features
Check existing issues before creating new ones
Include system details and error logs
Star ⭐ this repo if you find it helpful!