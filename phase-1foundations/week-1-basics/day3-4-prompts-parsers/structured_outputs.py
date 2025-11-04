"""
Day 3-4: Structured Outputs with Pydantic
Learning: Type-safe parsing, validation, and error handling
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json

llm = OllamaLLM(model="command-r", temperature=0.3)  # Lower temp for structured output

# ============================================================================
# EXAMPLE 1: Basic Pydantic Model
# ============================================================================
print("=== EXAMPLE 1: BASIC PYDANTIC MODEL ===\n")

class IncidentReport(BaseModel):
    """Schema for incident report extraction"""
    incident_id: str = Field(description="Unique identifier for the incident")
    severity: str = Field(description="Severity level: CRITICAL, HIGH, MEDIUM, LOW")
    description: str = Field(description="Brief description of the incident")
    affected_systems: List[str] = Field(description="List of affected systems")
    estimated_impact: str = Field(description="Business impact assessment")
    
parser = PydanticOutputParser(pydantic_object=IncidentReport)

prompt = PromptTemplate(
    input_variables=["incident_text"],
    template="""Extract structured information from this incident report:

{incident_text}

{format_instructions}

Return only valid JSON matching the schema.""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

incident_text = """
INC-2024-10234: Production payment gateway experienced intermittent 
timeout errors affecting approximately 1500 transactions during peak hours. 
Payment API, Transaction DB, and Notification Service all showed degraded 
performance. Estimated revenue impact: $75,000 in failed transactions. 
Customer support received 200+ complaints.
"""

try:
    result = chain.invoke({"incident_text": incident_text})
    print(f"Parsed Incident Report:")
    print(f"  ID: {result.incident_id}")
    print(f"  Severity: {result.severity}")
    print(f"  Description: {result.description}")
    print(f"  Affected Systems: {', '.join(result.affected_systems)}")
    print(f"  Impact: {result.estimated_impact}\n")
except Exception as e:
    print(f"Parsing error: {e}\n")

print("="*100 + "\n")

# ============================================================================
# EXAMPLE 2: Nested Pydantic Models
# ============================================================================
print("=== EXAMPLE 2: NESTED MODELS ===\n")

class SLAMetrics(BaseModel):
    """SLA performance metrics"""
    target_response_time_ms: int = Field(description="Target response time in milliseconds")
    actual_response_time_ms: int = Field(description="Actual response time in milliseconds")
    breach_status: str = Field(description="SLA status: COMPLIANT, BREACHED, AT_RISK")
    breach_duration_minutes: Optional[int] = Field(description="Duration of SLA breach in minutes")

class SystemHealth(BaseModel):
    """Complete system health report"""
    system_name: str = Field(description="Name of the system")
    availability_percentage: float = Field(description="System availability as percentage")
    sla_metrics: SLAMetrics = Field(description="SLA performance data")
    error_count_24h: int = Field(description="Number of errors in last 24 hours")
    recommendation: str = Field(description="Recommended action based on metrics")

parser2 = PydanticOutputParser(pydantic_object=SystemHealth)

prompt2 = PromptTemplate(
    input_variables=["system_data"],
    template="""Analyze this system monitoring data and extract structured information:

{system_data}

{format_instructions}

Provide complete JSON matching the schema.""",
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

chain2 = prompt2 | llm | parser2

system_data = """
Payment Processing Service showed 99.2% uptime over the last week. 
API response times averaged 850ms against a 500ms target, representing 
an SLA breach that lasted approximately 45 minutes during peak load. 
System logged 234 errors in the past 24 hours, mostly timeout exceptions. 
Immediate scaling of compute resources recommended.
"""

try:
    result2 = chain2.invoke({"system_data": system_data})
    print(f"System Health Report:")
    print(f"  System: {result2.system_name}")
    print(f"  Availability: {result2.availability_percentage}%")
    print(f"  SLA Status: {result2.sla_metrics.breach_status}")
    print(f"  Target Response: {result2.sla_metrics.target_response_time_ms}ms")
    print(f"  Actual Response: {result2.sla_metrics.actual_response_time_ms}ms")
    print(f"  24h Errors: {result2.error_count_24h}")
    print(f"  Recommendation: {result2.recommendation}\n")
except Exception as e:
    print(f"Parsing error: {e}\n")

print("="*100 + "\n")

# ============================================================================
# EXAMPLE 3: JsonOutputParser (More Flexible)
# ============================================================================
print("=== EXAMPLE 3: JSON OUTPUT PARSER ===\n")

json_prompt = PromptTemplate(
    input_variables=["transaction_data"],
    template="""Analyze this payment transaction data and return JSON with these fields:
- transaction_id (string)
- amount_usd (number)
- status (string: success, failed, pending)
- failure_reason (string or null)
- retry_recommended (boolean)
- priority_level (string: high, medium, low)

Transaction Data: {transaction_data}

Return ONLY valid JSON, no other text:"""
)

json_parser = JsonOutputParser()
chain3 = json_prompt | llm | json_parser

transaction_data = """
TXN-98765: Wire transfer of $125,000 from Account A to Account B 
failed due to 'insufficient funds verification timeout'. 
Transaction initiated 3 times, all failed. Customer is VIP tier. 
Current account balance shows sufficient funds.
"""

try:
    result3 = chain3.invoke({"transaction_data": transaction_data})
    print(f"Parsed Transaction Analysis:")
    print(json.dumps(result3, indent=2))
    print()
except Exception as e:
    print(f"JSON parsing error: {e}\n")

print("="*100 + "\n")

# ============================================================================
# EXAMPLE 4: Error Handling & Fallback
# ============================================================================
print("=== EXAMPLE 4: ERROR HANDLING ===\n")

class FailedClientInteraction(BaseModel):
    """Schema for failed client interaction analysis"""
    interaction_type: str = Field(description="Type: API_CALL, UI_ACTION, BATCH_JOB")
    failure_count: int = Field(description="Number of failures")
    impact_score: float = Field(description="Impact score from 0-10")
    root_cause: str = Field(description="Identified root cause")
    mitigation_steps: List[str] = Field(description="List of mitigation steps")

parser4 = PydanticOutputParser(pydantic_object=FailedClientInteraction)

prompt4 = PromptTemplate(
    input_variables=["fci_data"],
    template="""Analyze this Failed Client Interaction data:

{fci_data}

{format_instructions}

Return valid JSON only.""",
    partial_variables={"format_instructions": parser4.get_format_instructions()}
)

chain4 = prompt4 | llm

fci_data = """
API endpoint /api/v2/payments/process returned 503 Service Unavailable 
for 450 client requests over a 15-minute window. Load balancer logs show 
backend pool exhaustion. High business impact affecting real-time payments. 
Immediate actions: scale backend, implement circuit breaker, add request queuing.
"""

def safe_parse(chain, parser, input_data, max_retries=2):
    """Parse with fallback and retries"""
    for attempt in range(max_retries):
        try:
            raw_output = chain.invoke(input_data)
            parsed = parser.parse(raw_output)
            print(f"✅ Successfully parsed on attempt {attempt + 1}")
            return parsed
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("⚠️  All parsing attempts failed. Returning raw output.")
                return raw_output
    return None

result4 = safe_parse(chain4, parser4, {"fci_data": fci_data})

if isinstance(result4, FailedClientInteraction):
    print(f"\nParsed FCI Analysis:")
    print(f"  Type: {result4.interaction_type}")
    print(f"  Failures: {result4.failure_count}")
    print(f"  Impact Score: {result4.impact_score}/10")
    print(f"  Root Cause: {result4.root_cause}")
    print(f"  Mitigations: {', '.join(result4.mitigation_steps)}")
else:
    print(f"\nRaw output (parsing failed):\n{result4}")

print("\n" + "="*100 + "\n")
