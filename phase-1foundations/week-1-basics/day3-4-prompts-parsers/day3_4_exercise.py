"""
Day 3-4: Hands-On Exercise
Build a Production Incident Data Extractor
Real-world application: Parse unstructured incident reports into structured data
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json

llm = OllamaLLM(model="command-r", temperature=0.3)

# ============================================================================
# Define Comprehensive Data Schema
# ============================================================================

class ImpactMetrics(BaseModel):
    """Business and technical impact metrics"""
    affected_user_count: Optional[int] = Field(description="Number of affected users")
    failed_transactions: Optional[int] = Field(description="Number of failed transactions")
    revenue_impact_usd: Optional[float] = Field(description="Estimated revenue impact in USD")
    customer_complaints: Optional[int] = Field(description="Number of customer complaints")
    sla_breach_minutes: Optional[int] = Field(description="Duration of SLA breach in minutes")

class RootCauseAnalysis(BaseModel):
    """Root cause analysis details"""
    primary_cause: str = Field(description="Primary root cause")
    contributing_factors: List[str] = Field(description="Contributing factors")
    affected_components: List[str] = Field(description="Affected system components")

class ResolutionPlan(BaseModel):
    """Incident resolution plan"""
    immediate_actions: List[str] = Field(description="Immediate actions taken")
    preventive_measures: List[str] = Field(description="Preventive measures for future")
    estimated_resolution_hours: float = Field(description="Estimated hours to full resolution")

class ProductionIncident(BaseModel):
    """Complete production incident report"""
    incident_id: str = Field(description="Unique incident identifier")
    title: str = Field(description="Brief incident title")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, LOW")
    category: str = Field(description="Incident category")
    description: str = Field(description="Detailed description")
    impact_metrics: ImpactMetrics = Field(description="Business and technical impact")
    root_cause: RootCauseAnalysis = Field(description="Root cause analysis")
    resolution: ResolutionPlan = Field(description="Resolution plan")
    priority_score: int = Field(description="Priority score from 1-10")

# ============================================================================
# Create Parser and Prompt
# ============================================================================

parser = PydanticOutputParser(pydantic_object=ProductionIncident)

# Few-shot examples for better extraction
examples = [
    {
        "raw_text": """
        INC-2024-1001: Payment Gateway Timeout
        Severity: CRITICAL
        Between 2 PM - 4 PM, payment gateway experienced widespread timeout errors.
        3,500 users affected, 1,200 transactions failed, estimated $250,000 revenue impact.
        850 support tickets opened. SLA breached by 90 minutes.
        Root cause: Database connection pool exhausted due to long-running queries from 
        new reporting feature deployed yesterday. Connection leak in reporting module.
        Immediate fix: Rolled back reporting feature, restarted connection pool.
        Prevention: Add connection pooling monitoring, implement query timeout limits,
        add circuit breakers. Estimated full resolution: 24 hours for code fixes.
        """,
        "structured": "Focus on extracting all numeric metrics, identifying system components, and listing actionable items"
    }
]

example_template = PromptTemplate(
    input_variables=["raw_text", "structured"],
    template="Raw Incident Text:\n{raw_text}\n\nExtraction Note: {structured}"
)

main_prompt = PromptTemplate(
    input_variables=["incident_text"],
    template="""You are an expert incident analyst for banking production systems.

Extract comprehensive structured data from this incident report:

{incident_text}

{format_instructions}

Return ONLY valid JSON matching the schema. Be thorough in extracting all metrics and details.""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = main_prompt | llm | parser

# ============================================================================
# Test Cases: Real-World Incident Reports
# ============================================================================

test_incidents = [
    """
    INC-2024-2045 - Mobile Banking App Crash Loop
    SEVERITY: HIGH
    
    Starting at 8:30 AM, mobile banking app (iOS and Android) entered crash loop 
    on launch for approximately 6,000 users (15% of mobile user base). Users unable 
    to access accounts, check balances, or make transfers. Zero revenue impact 
    (no transaction fees lost) but significant reputation damage. Customer support 
    received 420 calls and 200+ App Store negative reviews within 2 hours.
    
    SLA: Availability target 99.95% breached by 135 minutes.
    
    ROOT CAUSE INVESTIGATION:
    - New app version 3.2.1 deployed last night with updated SSL certificate pinning
    - Certificate validation logic contained bug causing verification failure
    - Backend API servers were functioning normally
    - Issue isolated to client-side certificate validation code
    
    RESOLUTION:
    Immediate: Emergency hotfix version 3.2.2 deployed to app stores (expedited review)
    Actions Taken:
    1. Rolled back SSL pinning changes
    2. Implemented phased rollout process for app updates
    3. Added client-side error telemetry
    4. Created app crash monitoring dashboard
    
    Prevention:
    1. Mandatory QA testing of SSL/certificate changes on multiple devices
    2. Implement feature flags for critical security features
    3. Staged rollout (5% -> 25% -> 100%) for app updates
    4. Add circuit breaker for certificate validation failures
    
    Timeline: Hotfix live in 6 hours, full monitoring implementation 48 hours
    """,
    
    """
    INC-2024-2156 - International Wire Transfer Processing Failure
    PRIORITY: CRITICAL - REVENUE IMPACTING
    
    Incident Duration: Oct 23, 2024, 10:15 AM - 1:45 PM (3.5 hours)
    
    IMPACT:
    International wire transfer processing completely halted. 287 high-value 
    wire transfers (total value $45.7 million) stuck in pending state. 
    23 enterprise customers (Fortune 500 companies) affected. Direct revenue 
    impact estimated at $180,000 in transfer fees. Regulatory reporting 
    obligation breach (same-day settlement requirement). 15 executive 
    escalations received.
    
    SLA Breach: 99.99% uptime target = 4.32 minutes/month allowance. 
    This incident consumed 210 minutes (4,861% of monthly budget).
    
    TECHNICAL DETAILS:
    System: International Payment Gateway (IPG)
    Components Affected: 
    - SWIFT message processor
    - Currency conversion service  
    - Compliance validation engine
    - Transaction reconciliation system
    
    ROOT CAUSE:
    Compliance validation engine deadlock caused by concurrent processing of 
    new sanctions list update (10,000+ new entities) while processing high 
    transaction volume. Database locks cascaded to dependent services.
    
    IMMEDIATE RESPONSE:
    1. Killed deadlocked database sessions (10:45 AM)
    2. Temporarily bypassed sanctions check with manual review process
    3. Processed stuck transactions in priority order (enterprise first)
    4. Engaged external compliance auditor for manual sanctions screening
    5. War room established with CTO, CISO, and Head of Compliance
    
    PERMANENT FIXES:
    1. Implement asynchronous sanctions list updates during low-traffic window
    2. Add database query timeout limits (max 30 seconds)
    3. Create separate read-replicas for compliance checks
    4. Implement circuit breaker pattern for compliance service
    5. Add transaction queuing with priority lanes
    6. Regulatory reporting automation for breach notifications
    
    ESTIMATED EFFORT:
    Critical fixes: 72 hours
    Full implementation of preventive measures: 3 weeks
    Infrastructure changes (read replicas): 2 weeks
    
    CUSTOMER COMMUNICATION:
    - Immediate notification to all affected customers
    - Waived all transfer fees for affected transactions ($180K)
    - Provided detailed RCA report to top 10 enterprise customers
    - Scheduled executive calls with all Fortune 500 clients
    """,
    
    """
    INC-2024-1987 - Batch Processing Performance Degradation
    Severity: MEDIUM (Internal SLA Only)
    
    Overnight batch job processing window exceeded by 4 hours. End-of-day 
    reconciliation reports delayed from 6 AM target to 10 AM actual delivery.
    
    No customer-facing impact (reports used internally only). 5 downstream 
    analytics dashboards showed stale data. Treasury team delayed daily 
    liquidity analysis by 3 hours.
    
    Affected Systems: ETL pipeline, data warehouse, reporting engine
    
    Cause: Database table statistics outdated leading to poor query plans. 
    One reconciliation query scanning 500M rows instead of using index 
    (expected 50K rows).
    
    Fix: Updated table statistics, added index on transaction_date column, 
    rewrote query with better hints. Batch job now completing in 3.5 hours 
    (was taking 8 hours).
    
    Actions:
    1. Schedule automatic statistics updates weekly
    2. Add query performance monitoring with alerting
    3. Review all batch queries for optimization opportunities
    
    Effort: Core fix completed, monitoring setup needs 16 hours work.
    """
]

# ============================================================================
# Process and Display Results
# ============================================================================

def safe_extract(incident_text, attempt_num=1, max_attempts=2):
    """Extract with retry logic"""
    try:
        result = chain.invoke({"incident_text": incident_text})
        return result, True
    except Exception as e:
        print(f"   ⚠️  Extraction attempt {attempt_num} failed: {str(e)[:100]}")
        if attempt_num < max_attempts:
            print(f"   🔄 Retrying...")
            return safe_extract(incident_text, attempt_num + 1, max_attempts)
        return None, False

print("="*100)
print("PRODUCTION INCIDENT DATA EXTRACTION SYSTEM")
print("="*100 + "\n")

for i, incident in enumerate(test_incidents, 1):
    print(f"\n{'='*100}")
    print(f"TEST CASE {i}")
    print(f"{'='*100}\n")
    
    print(f"Raw Incident Report (first 200 chars):\n{incident[:200]}...\n")
    print("🔄 Extracting structured data...\n")
    
    result, success = safe_extract(incident)
    
    if success and result:
        print("✅ EXTRACTION SUCCESSFUL\n")
        print(f"📋 INCIDENT SUMMARY")
        print(f"   ID: {result.incident_id}")
        print(f"   Title: {result.title}")
        print(f"   Severity: {result.severity}")
        print(f"   Category: {result.category}")
        print(f"   Priority Score: {result.priority_score}/10\n")
        
        print(f"📊 IMPACT METRICS")
        if result.impact_metrics.affected_user_count:
            print(f"   Affected Users: {result.impact_metrics.affected_user_count:,}")
        if result.impact_metrics.failed_transactions:
            print(f"   Failed Transactions: {result.impact_metrics.failed_transactions:,}")
        if result.impact_metrics.revenue_impact_usd:
            print(f"   Revenue Impact: ${result.impact_metrics.revenue_impact_usd:,.2f}")
        if result.impact_metrics.customer_complaints:
            print(f"   Customer Complaints: {result.impact_metrics.customer_complaints}")
        if result.impact_metrics.sla_breach_minutes:
            print(f"   SLA Breach: {result.impact_metrics.sla_breach_minutes} minutes\n")
        
        print(f"🔍 ROOT CAUSE ANALYSIS")
        print(f"   Primary Cause: {result.root_cause.primary_cause}")
        print(f"   Contributing Factors:")
        for factor in result.root_cause.contributing_factors:
            print(f"      • {factor}")
        print(f"   Affected Components: {', '.join(result.root_cause.affected_components)}\n")
        
        print(f"🛠️  RESOLUTION PLAN")
        print(f"   Immediate Actions:")
        for action in result.resolution.immediate_actions[:3]:  # Show first 3
            print(f"      • {action}")
        print(f"   Preventive Measures:")
        for measure in result.resolution.preventive_measures[:3]:  # Show first 3
            print(f"      • {measure}")
        print(f"   Estimated Resolution: {result.resolution.estimated_resolution_hours} hours\n")
        
        # Save to JSON file
        output_file = f"incident_{result.incident_id.replace('-', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        print(f"💾 Saved to: {output_file}")
        
    else:
        print("❌ EXTRACTION FAILED")
        print("   Manual review required\n")
    
    print(f"\n{'='*100}\n")

print("\n🎯 EXTRACTION COMPLETE")
print(f"Successfully processed {len(test_incidents)} incident reports")
print("Check generated JSON files for full structured data\n")
