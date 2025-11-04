"""
Day 3-4: Advanced Prompt Engineering
Learning: Multiple variables, roles, context, and systematic instructions
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="command-r", temperature=0.7)

# ============================================================================
# TECHNIQUE 1: Role-Based Prompting
# ============================================================================
print("=== TECHNIQUE 1: ROLE-BASED PROMPTING ===\n")

role_prompt = PromptTemplate(
    input_variables=["role", "task", "domain"],
    template="""You are a {role} with 15 years of experience in {domain}.

Task: {task}

Provide a comprehensive, expert-level response using industry terminology.
Format your response with clear sections and bullet points where appropriate.

Response:"""
)

# Test with banking domain
chain = role_prompt | llm | StrOutputParser()
result = chain.invoke({
    "role": "Senior Production Support Engineer",
    "domain": "banking payments and transaction processing",
    "task": "Explain the incident management workflow for a critical payment processing failure affecting 1000+ transactions"
})
print(f"Role-based response:\n{result}\n")
print("="*100 + "\n")

# ============================================================================
# TECHNIQUE 2: Context + Constraints
# ============================================================================
print("=== TECHNIQUE 2: CONTEXT + CONSTRAINTS ===\n")

constrained_prompt = PromptTemplate(
    input_variables=["context", "question", "max_words", "tone"],
    template="""Context: {context}

Question: {question}

Constraints:
- Maximum {max_words} words
- Tone: {tone}
- Include at least one concrete example
- Use bullet points for key points

Answer:"""
)

chain2 = constrained_prompt | llm | StrOutputParser()
result2 = chain2.invoke({
    "context": "A wholesale banking platform processing $5B daily in international payments",
    "question": "What metrics should we monitor to predict payment processing failures?",
    "max_words": "150",
    "tone": "technical and precise"
})
print(f"Constrained response:\n{result2}\n")
print("="*100 + "\n")

# ============================================================================
# TECHNIQUE 3: Multi-Step Instructions
# ============================================================================
print("=== TECHNIQUE 3: MULTI-STEP INSTRUCTIONS ===\n")

multistep_prompt = PromptTemplate(
    input_variables=["scenario", "requirement"],
    template="""Scenario: {scenario}

Requirement: {requirement}

Follow these steps in your response:
1. ANALYZE: Identify the core problem and root causes
2. PRIORITIZE: Rank issues by severity and business impact
3. RECOMMEND: Provide 3 specific, actionable solutions
4. TIMELINE: Suggest implementation order and estimated effort

Use clear headers for each step.

Analysis:"""
)

chain3 = multistep_prompt | llm | StrOutputParser()
result3 = chain3.invoke({
    "scenario": "Production incident: Payment API latency increased from 200ms to 5000ms affecting all customer transactions",
    "requirement": "Develop an immediate action plan and long-term prevention strategy"
})
print(f"Multi-step response:\n{result3}\n")
print("="*100 + "\n")

# ============================================================================
# TECHNIQUE 4: Comparison Prompts
# ============================================================================
print("=== TECHNIQUE 4: COMPARISON PROMPTS ===\n")

comparison_prompt = PromptTemplate(
    input_variables=["option_a", "option_b", "criteria"],
    template="""Compare the following two approaches based on: {criteria}

Option A: {option_a}

Option B: {option_b}

Provide a structured comparison:
1. Pros and Cons for each option
2. Best use cases for each
3. Your recommendation with justification

Comparison:"""
)

chain4 = comparison_prompt | llm | StrOutputParser()
result4 = chain4.invoke({
    "option_a": "Monolithic architecture for payment processing",
    "option_b": "Microservices architecture for payment processing",
    "criteria": "scalability, reliability, maintenance complexity, and incident response"
})
print(f"Comparison response:\n{result4}\n")
print("="*100 + "\n")

# ============================================================================
# TECHNIQUE 5: Template with Examples (Inline Few-Shot)
# ============================================================================
print("=== TECHNIQUE 5: INLINE FEW-SHOT EXAMPLES ===\n")

fewshot_inline_prompt = PromptTemplate(
    input_variables=["incident"],
    template="""You are an incident severity classifier for banking systems.

Examples:
Incident: "Database connection pool exhausted, 50% of API requests failing"
Severity: CRITICAL - Immediate revenue impact, customer-facing
Action: Page on-call engineer, initiate war room

Incident: "Background batch job delayed by 2 hours, no customer impact"
Severity: LOW - Internal only, scheduled maintenance window available
Action: Create ticket for next sprint, monitor

Incident: "SSL certificate expires in 3 days for admin portal"
Severity: MEDIUM - Non-customer facing but requires timely action
Action: Assign to security team, set reminder

Now classify:
Incident: {incident}
Severity:"""
)

chain5 = fewshot_inline_prompt | llm | StrOutputParser()
result5 = chain5.invoke({
    "incident": "Payment reconciliation system showing 0.5% discrepancy in transaction amounts"
})
print(f"Few-shot classification:\n{result5}\n")
print("="*100 + "\n")
