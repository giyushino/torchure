# torchure
pure pytorch training stack with minimal deps

want to support training AR LLMs, DLLMs, 
continuous diffusiion LLMs, etc

focus on AR for now

## Installation
installation depends on whether or not you have
uv installed on your machine. In theory you
don't need uv, but it is a lot easier to use

```bash
conda create -n fresh python==3.14
conda activate fresh
pip install uv
uv pip install -e .
```
