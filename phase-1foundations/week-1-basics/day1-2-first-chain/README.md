# Day 1-2: First LangChain Chain

## 🎯 Learning Objectives
- Set up LangChain development environment
- Understand core components: LLM, PromptTemplate, Chain
- Build first chain using LCEL (LangChain Expression Language)
- Create domain-specific assistant (Banking Q&A)

## 📋 Prerequisites
- ✅ Ollama installed
- ✅ Cohere Command R model pulled
- ✅ Python 3.9+ with virtual environment

## 🛠️ Setup

### Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### Verify Ollama
ollama list # Should show command-r
ollama run command-r "Hello" # Test model


## 📚 Core Concepts Learned

### 1. LLM Wrapper
- `OllamaLLM`: Wrapper for local Ollama models
- Parameters: `model`, `temperature`
- Direct invocation: `llm.invoke("prompt")`

### 2. Prompt Templates
- `PromptTemplate`: Parameterized prompt strings
- `input_variables`: Dynamic placeholders
- `.format()`: Fill in variables

### 3. Output Parsers
- `StrOutputParser`: Returns plain string
- `JsonOutputParser`: Parses JSON responses
- Transforms LLM output into usable format

### 4. LCEL (LangChain Expression Language)
- Chain components with pipe operator: `|`
- Flow: `prompt | llm | parser`
- Clean, readable syntax

## 📝 Scripts Overview

| Script | Purpose | Key Learning |
|--------|---------|--------------|
| `first_chain.py` | Basic chain implementation | Prompt → LLM → Output flow |
| `components_explained.py` | Deep dive into each component | Understanding building blocks |
| `day1_exercise.py` | Banking domain Q&A | Practical application |

## 🧪 Running the Code

Test basic chain
python first_chain.py

Understand components
python components_explained.py

Hands-on exercise
python day1_exercise.py


## 📊 Results & Observations

### Performance Metrics
- **Model**: Cohere Command R (35B)
- **Hardware**: M4 Pro, 24GB RAM
- **Average response time**: [Add your observation]
- **Memory usage**: [Monitor Activity Monitor]

### Key Observations
- [Add what you noticed about response quality]
- [Add any errors or issues encountered]
- [Add insights about prompt engineering]

## 🔑 Key Takeaways
1. LangChain chains are composable pipelines
2. LCEL syntax (pipe operator) creates clean data flows
3. Prompt engineering significantly affects output quality
4. Local models (Command R) provide fast, private inference

## 🚀 Next Steps
- Day 3-4: Advanced prompt engineering and output parsing
- Explore different temperature settings
- Experiment with prompt variations

## 🐛 Troubleshooting

### Ollama Connection Error
Ensure Ollama is running
ollama serve # If not running as service


### Import Errors
Reinstall packages
pip install --upgrade langchain langchain-community langchain-ollama


## 📂 File Structure
day1-2-first-chain/
├── README.md
├── requirements.txt
├── first_chain.py
├── components_explained.py
└── day1_exercise.py
