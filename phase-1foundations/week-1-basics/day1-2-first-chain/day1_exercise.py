"""
Day 1-2: Hands-On Exercise
Build a Banking Domain Q&A Assistant
Based on your domain expertise in banking and payments
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = OllamaLLM(model="command-r", temperature=0.7)

# Banking domain prompt template
banking_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a banking and payments domain expert assistant. You specialize in:
- Production support for banking applications
- Payment processing systems
- Incident management and SLA compliance
- Banking regulations and compliance

Context: {context}

Question: {question}

Provide a clear, technical answer based on your banking domain expertise:"""
)

# Create the chain
banking_chain = banking_prompt | llm | StrOutputParser()

# Test questions based on your banking experience
test_questions = [
    {
        "question": "What is the impact of a failed client interaction in a payment processing system?",
        "context": "A wholesale banking platform processing high-value transactions"
    },
    {
        "question": "How should we prioritize incidents with SLA breaches?",
        "context": "Production support team managing L2/L3 incidents"
    },
    {
        "question": "What are the key components of a real-time payment system?",
        "context": "Building a new payment processing microservice"
    },
    {
        "question": "Explain the role of monitoring in preventing production incidents",
        "context": "SRE practices for a banking application"
    }
]

print("=== Banking Domain Q&A Assistant ===\n")
for i, qa in enumerate(test_questions, 1):
    print(f"Question {i}: {qa['question']}")
    print(f"Context: {qa['context']}")
    
    response = banking_chain.invoke({
        "question": qa['question'],
        "context": qa['context']
    })
    
    print(f"\nAnswer:\n{response}")
    print("\n" + "="*100 + "\n")

# Interactive mode (optional)
print("=== Interactive Mode ===")
print("Enter your banking/payments questions (type 'exit' to quit)")

while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == 'exit':
        break
    
    context = input("Context (press Enter for general): ")
    if not context:
        context = "General banking and payments domain"
    
    answer = banking_chain.invoke({
        "question": user_question,
        "context": context
    })
    
    print(f"\nAnswer: {answer}\n")
