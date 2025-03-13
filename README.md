# Advanced Wikipedia-Ollama Query System

This tool combines the power of Ollama language models with Wikipedia to provide comprehensive answers to complex queries. It extracts key topics from user questions, searches Wikipedia for relevant information, and synthesizes well-structured responses.

### Features

- Intelligent topic extraction and refinement
- Automatic Wikipedia search optimization
- Multi-threaded information gathering
- Response synthesis and verification
- Markdown or plain text output options

### Requirements

- Python 3.6+
- Ollama installed and running

### Installation

```bash
# Install dependencies
pip install ollama wikipedia beautifulsoup4
```

### Usage

```bash
python advanced_wiki_ollama.py --model gemma3:12b
```

### Command Line Options

- `--model`: Ollama model to use (default: gemma3:12b)
- `--log-file`: Log file path (default: advanced_wiki_ollama.log)
- `--console-log`: Enable console logging
- `--max-topics`: Maximum number of topics to process (default: 7)
- `--plain-text`: Output plain text instead of markdown
- `--timeout`: Timeout in seconds for LLM requests (default: 60)

### Example

```
> What are the key principles of REST API design?

Processing your query. This may take a few moments...

Extracting key topics...
Evaluating topic granularity...
Refining topics...
Optimizing topics for Wikipedia search...
Scoring topic relevance...
Gathering information on 5 topics...
Synthesizing information...
Generating response...
Verifying response quality...

Answer:
[Comprehensive answer about REST API design principles]

Processing time: 45.23 seconds
```

### License

MIT
