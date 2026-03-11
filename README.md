# Fraud Intelligence Copilot

AI-powered fraud detection dashboard combining ML scoring with LLM explanations via local Ollama.

## Architecture

- **Backend**: FastAPI + scikit-learn GradientBoosting classifier
- **LLM**: Ollama (Llama 3) for natural language fraud explanations
- **Frontend**: React + Vite dashboard
- **Dataset**: Kaggle Credit Card Fraud Detection

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Ollama (`brew install ollama`)

### 1. Install Ollama & pull model
```bash
brew install ollama
ollama pull llama3
```

### 2. Download dataset
Download [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place in `data/`

### 3. Train model
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m backend.train --data data/creditcard.csv
```

### 4. Run
```bash
# Terminal 1 — API
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Ollama
ollama serve

# Terminal 3 — Frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:3000
