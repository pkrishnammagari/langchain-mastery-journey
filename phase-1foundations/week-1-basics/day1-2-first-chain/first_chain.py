"""
Day 1-2: First LangChain Chain
Learning: Basic prompt template, LLM wrapper, and chain execution
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Ollama with Command R (running locally)
llm = OllamaLLM(
    model="command-r",
    temperature=0.7,
)

# Test basic LLM call
print("=== Testing Basic LLM Call ===")
response = llm.invoke("Explain what LangChain is in one sentence.")
print(f"Response: {response}\n")

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="You are a helpful AI assistant. Explain {topic} in simple terms suitable for a beginner. Keep it under 100 words."
)

# Build a chain using LCEL (LangChain Expression Language)
chain = prompt | llm | StrOutputParser()

# Execute the chain
print("=== Testing Chain with Prompt Template ===")
result = chain.invoke({"topic": "Retrieval Augmented Generation (RAG)"})
print(f"Result: {result}\n")

# Multiple invocations with different topics
topics = [
    "Large Language Models",
    "Vector Databases",
    "Prompt Engineering"
]

print("=== Testing Multiple Invocations ===")
for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"\nTopic: {topic}")
    print(f"Explanation: {result}")
    print("-" * 80)
