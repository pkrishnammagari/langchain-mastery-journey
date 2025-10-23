"""
Day 1-2: Understanding LangChain Components
Learning: Deep dive into each component
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# Component 1: LLM Wrapper
print("=== 1. LLM WRAPPER ===")
llm = OllamaLLM(model="command-r", temperature=0.3)
print("LLM initialized with model: command-r")
print("Temperature: 0.3 (more deterministic)\n")

# Component 2: Simple Prompt Template
print("=== 2. SIMPLE PROMPT TEMPLATE ===")
simple_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question concisely: {question}"
)
formatted = simple_prompt.format(question="What is LangChain?")
print(f"Formatted prompt: {formatted}\n")

# Component 3: Advanced Prompt Template with Multiple Variables
print("=== 3. ADVANCED PROMPT TEMPLATE ===")
advanced_prompt = PromptTemplate(
    input_variables=["role", "task", "context"],
    template="""You are a {role}.

Task: {task}

Context: {context}

Provide your response:"""
)
formatted_advanced = advanced_prompt.format(
    role="Python Developer",
    task="Explain the benefits of using virtual environments",
    context="A beginner just installed Python and wants to start a project"
)
print(f"Formatted advanced prompt:\n{formatted_advanced}\n")

# Component 4: String Output Parser (default)
print("=== 4. STRING OUTPUT PARSER ===")
string_chain = simple_prompt | llm | StrOutputParser()
result = string_chain.invoke({"question": "What is a vector database?"})
print(f"Type: {type(result)}")
print(f"Result: {result}\n")

# Component 5: Structured Output with JSON Parser
print("=== 5. STRUCTURED OUTPUT (JSON) ===")

class ConceptExplanation(BaseModel):
    """Schema for structured output"""
    concept: str = Field(description="The concept being explained")
    definition: str = Field(description="A brief definition")
    use_case: str = Field(description="A practical use case")
    
json_prompt = PromptTemplate(
    input_variables=["concept"],
    template="""Explain the following concept and provide output as JSON with these fields:
- concept: the name of the concept
- definition: a brief definition (max 50 words)
- use_case: a practical use case (max 50 words)

Concept: {concept}

Return only valid JSON, nothing else."""
)

json_chain = json_prompt | llm | JsonOutputParser(pydantic_object=ConceptExplanation)

try:
    structured_result = json_chain.invoke({"concept": "RAG (Retrieval Augmented Generation)"})
    print(f"Type: {type(structured_result)}")
    print(f"Structured Result:")
    print(f"  Concept: {structured_result.get('concept')}")
    print(f"  Definition: {structured_result.get('definition')}")
    print(f"  Use Case: {structured_result.get('use_case')}\n")
except Exception as e:
    print(f"Error parsing JSON: {e}")
    print("Note: LLMs may not always return perfect JSON. We'll handle this in later days.\n")

# Component 6: Chain Composition with LCEL
print("=== 6. CHAIN COMPOSITION (LCEL) ===")
print("LCEL uses the pipe operator (|) to chain components:")
print("prompt | llm | output_parser")
print("\nThis creates a data flow: input -> prompt formatting -> LLM -> parsing -> output")
