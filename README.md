# 🔬 Advanced AI Reasoning Research System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/ai-reasoning-system?style=social)](https://github.com/your-username/ai-reasoning-system)

An open-source research platform that implements cutting-edge AI reasoning methodologies including **Tree of Thoughts**, **Constitutional AI**, and **multi-agent debate patterns**. Features a modern web interface, real-time streaming, and comprehensive analytics.

---

## 🎯 What This Project Does

- **Multi-Strategy Reasoning**: Apply different reasoning approaches to the same problem
- **Self-Critique System**: AI reviews and improves its own responses
- **Real-time Analytics**: Track reasoning depth, confidence, and performance metrics
- **Export & Documentation**: Save conversations as PDF, Markdown, or JSON
- **Production Ready**: Caching, rate limiting, error handling, and automatic backups

---

## 🚀 Quick Start (2 Minutes)

### Prerequisites

- Python **3.8+**
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/ai-reasoning-system.git
cd ai-reasoning-system

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# Launch system
python main.py
```

Open your browser to `http://localhost:7860` and start exploring!

---

## 📊 Reasoning Strategies

| Method | Description | Best For |
|--------|-------------|----------|
| **Tree of Thoughts** | Explores multiple reasoning paths systematically | Complex problems with multiple solutions |
| **Chain of Thought** | Step-by-step transparent reasoning | Mathematical problems, logic puzzles |
| **Self-Consistency** | Generates multiple answers and finds consensus | Factual questions, reliability important |
| **Reflexion** | Self-critique and iterative improvement | Creative writing, analysis tasks |
| **Multi-Agent Debate** | Presents multiple perspectives | Ethical dilemmas, policy questions |
| **Analogical Reasoning** | Finds similar problems and adapts solutions | Novel problems, innovation tasks |

---

## 🎥 Demo Features

### Real-time Interface

- **Streaming Responses**: Watch reasoning unfold in real-time
- **Live Metrics**: See inference time, tokens/second, reasoning depth
- **Interactive Controls**: Switch models, adjust temperature, enable critique
- **Modern Design**: Clean, responsive interface with dark theme

### Analytics Dashboard

- Session performance metrics
- Model usage distribution
- Cache hit rates
- Error tracking and retry statistics

### Export Options

- **PDF**: Professional reports with formatting
- **Markdown**: GitHub-friendly documentation
- **JSON**: Machine-readable data
- **Plain Text**: Simple conversation logs

---

## 🔧 Configuration

Key settings in `config.py`:

```python
MAX_HISTORY_LENGTH = 10          # Messages in context
CACHE_SIZE = 100                 # Cached responses
RATE_LIMIT_REQUESTS = 50         # Per minute
DEFAULT_TEMPERATURE = 0.7        # Creativity level
MAX_TOKENS = 4000                # Response length
```

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  Core Engine    │    │   Groq API      │
│                 │    │                 │    │                 │
│ • Chat Interface│◄──►│ • Reasoning     │◄──►│ • LLM Models    │
│ • Controls      │    │ • Caching       │    │ • Streaming     │
│ • Metrics       │    │ • Rate Limiting │    │ • Token Count   │
│ • Export        │    │ • Error Handling│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📈 Performance

- **Cold Start**: ~2 seconds
- **Time to First Token**: 0.3–1.2 seconds
- **Throughput**: Up to 100 tokens/second
- **Memory Usage**: ~100MB base + conversation history
- **Concurrent Users**: Limited by Groq rate limits (50 req/min)

---

## 🧪 Example Use Cases

### Research Analysis

```
User: "Analyze the impact of remote work on productivity"
System: Uses Tree of Thoughts to explore economic, psychological, and technological factors
```

### Code Review

```
User: "Review this Python function for errors"
System: Applies Chain of Thought to identify bugs, suggest improvements
```

### Creative Writing

```
User: "Write a story about AI consciousness"
System: Uses Reflexion to draft, critique, and refine the narrative
```

### Decision Making

```
User: "Should we implement a four-day work week?"
System: Multi-Agent Debate presents management and employee perspectives
```

---

## 📚 Research Foundation

Built on seminal papers:

- **Tree of Thoughts** (Yao et al., 2023) – Systematic exploration
- **Constitutional AI** (Bai et al., 2022) – Self-critique mechanisms
- **Chain of Thought** (Wei et al., 2022) – Transparent reasoning
- **Reflexion** (Shinn et al., 2023) – Iterative improvement
- **Self-Consistency** (Wang et al., 2022) – Consensus building

---

## 🔍 Project Structure

```
ai-reasoning-system/
├── main.py              # Gradio interface and event handlers
├── core.py              # Business logic and reasoning engine
├── config.py            # Configuration and constants
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
├── .env                 # API keys (created by user)
├── exports/             # Exported conversations
├── backups/             # Automatic backups
└── reasoning_system.log # Application logs
```

---

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=core
```

### Adding New Reasoning Mode

1. Add enum value in `ReasoningMode`
2. Add system prompt in `PromptEngine.SYSTEM_PROMPTS`
3. Add reasoning template in `PromptEngine.REASONING_PROMPTS`
4. Update UI choices in `main.py`

### Custom Models

Add to `ModelConfig` enum:

```python
CUSTOM_MODEL = ("custom-model-id", parameters, context_length, "Description")
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key Error | Check `.env` file format: `GROQ_API_KEY=gsk_...` |
| Rate Limit Hit | Wait 60 seconds or reduce request frequency |
| Memory Issues | Reduce `MAX_CONVERSATION_STORAGE` in config |
| PDF Export Fails | Install reportlab: `pip install reportlab` |
| Port Already in Use | Change port: `python main.py --port 7861` |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Academic Use

Perfect for:

- Final year projects
- Research demonstrations
- AI methodology studies
- Human-AI interaction experiments

### Citation

```bibtex
@software{ai_reasoning_system_2025,
  title = {Advanced AI Reasoning Research System},
  year = {2025},
  url = {https://github.com/your-username/ai-reasoning-system}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit Pull Request

---

## 📞 Support

- Create an [issue](https://github.com/your-username/ai-reasoning-system/issues) for bugs or features
- Check existing issues before creating new ones
- Include system details and error logs

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

Made with ❤️ by the AI Research Community

[Report Bug](https://github.com/your-username/ai-reasoning-system/issues) · [Request Feature](https://github.com/your-username/ai-reasoning-system/issues) · [Documentation](https://github.com/your-username/ai-reasoning-system/wiki)

</div>