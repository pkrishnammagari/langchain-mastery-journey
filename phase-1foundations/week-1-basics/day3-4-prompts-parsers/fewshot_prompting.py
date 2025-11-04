"""
Day 3-4: Few-Shot Prompting
Learning: Teaching LLMs by example
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="command-r", temperature=0.5)

# ============================================================================
# METHOD 1: FewShotPromptTemplate
# ============================================================================
print("=== METHOD 1: FEWSHOTPROMPTTEMPLATE ===\n")

# Define examples for incident classification
examples = [
    {
        "incident": "Database backup job failed on secondary server",
        "classification": "Priority: P3 (Low)\nCategory: Operational\nAction: Investigate during business hours\nEscalation: Not required"
    },
    {
        "incident": "Payment gateway API returning 500 errors for all transactions",
        "classification": "Priority: P1 (Critical)\nCategory: Service Outage\nAction: Immediate war room, page on-call\nEscalation: VP Engineering within 15 minutes"
    },
    {
        "incident": "Mobile app login slow, taking 5-8 seconds instead of 1-2 seconds",
        "classification": "Priority: P2 (High)\nCategory: Performance Degradation\nAction: Assign to performance team, investigate within 2 hours\nEscalation: Manager if not resolved in 4 hours"
    },
    {
        "incident": "Reporting dashboard shows stale data from 2 hours ago",
        "classification": "Priority: P3 (Medium)\nCategory: Data Issue\nAction: Check ETL jobs, create ticket\nEscalation: Team lead if data is 6+ hours old"
    }
]

# Create template for each example
example_template = PromptTemplate(
    input_variables=["incident", "classification"],
    template="Incident: {incident}\nClassification:\n{classification}"
)

# Create the few-shot prompt template
prefix = """You are an expert incident manager for a banking platform. 
Classify the following incidents using the same format as the examples below."""

suffix = """Incident: {incident}
Classification:"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix=prefix,
    suffix=suffix,
    input_variables=["incident"],
    example_separator="\n\n---\n\n"
)

chain = few_shot_prompt | llm | StrOutputParser()

# Test with new incidents
test_incidents = [
    "Customer data export API timing out for large datasets (>10MB)",
    "Production deployment failed health check, auto-rollback initiated",
    "Scheduled maintenance notification email not sent to customers"
]

for test_incident in test_incidents:
    print(f"Test Incident: {test_incident}")
    result = chain.invoke({"incident": test_incident})
    print(f"Model Classification:\n{result}\n")
    print("="*100 + "\n")

# ============================================================================
# METHOD 2: Dynamic Few-Shot (Banking Domain)
# ============================================================================
print("=== METHOD 2: DYNAMIC FEW-SHOT WITH DOMAIN EXPERTISE ===\n")

# Examples for SLA impact calculation
sla_examples = [
    {
        "scenario": "API downtime: 30 minutes during business hours, affecting 500 users",
        "calculation": """
SLA Calculation:
- Availability Target: 99.9% monthly
- Downtime Budget: 43.2 minutes/month
- Actual Downtime: 30 minutes
- SLA Status: WITHIN BUDGET (69% of budget used)
- Impact Score: 7/10 (High - business hours)
- Customer Credits: None required (within SLA)
- Action: Post-mortem required, monitor remaining budget
"""
    },
    {
        "scenario": "Batch processing delay: 6 hours, overnight window, no customer impact",
        "calculation": """
SLA Calculation:
- Batch Window Target: Complete by 6 AM
- Actual Completion: 12 PM (6 hours late)
- SLA Status: BREACHED (internal SLA only)
- Impact Score: 3/10 (Low - no customer facing impact)
- Customer Credits: None (internal process)
- Action: Optimize batch job, increase resources if recurring
"""
    },
    {
        "scenario": "Payment processing failure: 2 hours peak time, $500K transactions failed",
        "calculation": """
SLA Calculation:
- Availability Target: 99.99% for payment APIs
- Downtime Budget: 4.3 minutes/month
- Actual Downtime: 120 minutes
- SLA Status: SEVERELY BREACHED (2700% over budget)
- Impact Score: 10/10 (Critical - revenue and reputation)
- Customer Credits: Required per contract (penalty provisions)
- Action: Immediate RCA, executive notification, customer communication plan
"""
    }
]

sla_example_template = PromptTemplate(
    input_variables=["scenario", "calculation"],
    template="Scenario: {scenario}\n{calculation}"
)

sla_prefix = """You are an SLA compliance analyst for a banking platform.
Calculate SLA impact using the same methodology as these examples:"""

sla_suffix = """Scenario: {scenario}
SLA Calculation:"""

sla_few_shot = FewShotPromptTemplate(
    examples=sla_examples,
    example_prompt=sla_example_template,
    prefix=sla_prefix,
    suffix=sla_suffix,
    input_variables=["scenario"],
    example_separator="\n\n---\n\n"
)

sla_chain = sla_few_shot | llm | StrOutputParser()

# Test SLA calculations
test_scenarios = [
    "Mobile app crash: 45 minutes during lunch hour, 2000 users unable to check balances",
    "Wire transfer processing delayed: 3 hours, 50 high-value transactions ($10M total) pending"
]

for scenario in test_scenarios:
    print(f"Scenario: {scenario}\n")
    result = sla_chain.invoke({"scenario": scenario})
    print(f"SLA Analysis:\n{result}\n")
    print("="*100 + "\n")

# ============================================================================
# METHOD 3: Format-Learning Few-Shot
# ============================================================================
print("=== METHOD 3: FORMAT-LEARNING FEW-SHOT ===\n")

# Teach the model a specific output format
format_examples = [
    {
        "input": "Payment API latency increased",
        "output": "[METRIC] p95_latency | [THRESHOLD] 500ms | [ACTUAL] 2500ms | [STATUS] BREACH | [ACTION] Scale backend pods"
    },
    {
        "input": "Database connection pool at capacity",
        "output": "[METRIC] db_pool_usage | [THRESHOLD] 80% | [ACTUAL] 95% | [STATUS] CRITICAL | [ACTION] Increase pool size, check for leaks"
    },
    {
        "input": "Error rate spiking in authentication service",
        "output": "[METRIC] error_rate | [THRESHOLD] 1% | [ACTUAL] 8% | [STATUS] ALERT | [ACTION] Check auth provider status, enable fallback"
    }
]

format_example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

format_prefix = """Convert monitoring alerts into structured format following these examples:"""

format_suffix = """Input: {input}
Output:"""

format_few_shot = FewShotPromptTemplate(
    examples=format_examples,
    example_prompt=format_example_template,
    prefix=format_prefix,
    suffix=format_suffix,
    input_variables=["input"]
)

format_chain = format_few_shot | llm | StrOutputParser()

test_alerts = [
    "CPU utilization exceeding normal range on payment servers",
    "Failed login attempts increasing dramatically",
    "Transaction reconciliation batch job timing out"
]

for alert in test_alerts:
    result = format_chain.invoke({"input": alert})
    print(f"Alert: {alert}")
    print(f"Structured: {result}\n")

print("="*100 + "\n")
