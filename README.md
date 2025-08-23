# Asynchronous Reasoning Engine

A reasoning engine for decision support that tracks multiple contradictory hypotheses over noisy/asynchronous business data, reviving dormant theories with new evidence.

## Overview

The Asynchronous Reasoning Engine replicates human cognitive flexibility by maintaining multiple competing explanations simultaneously without forcing immediate resolution. Unlike traditional AI systems that require consistency maintenance, this engine embraces contradictory hypotheses as a feature rather than a logical error.

### Key Innovation: Single SLM with Token-Efficient Context Rewriting

Instead of complex multi-agent systems or large language models, we use a single Small Language Model (SLM) managing a rewritable context window filled with structured assertion tuples. This approach minimizes token consumption for long-running reasoning in business applications while maintaining the unified perspective that makes human reasoning effective.

Each piece of evidence and hypothesis becomes a timestamped tuple containing content, confidence level, status, source, and metadata - presented in a compact format that maximizes reasoning capability within constrained token budgets.

## Features

- **Living with Contradictions**: Maintains multiple competing hypotheses simultaneously
- **Token-Efficient Reasoning**: Structured tuples minimize context size for cost-effective SLM deployment
- **Deep Thought Processing**: Protected reasoning sessions with stable context snapshots
- **Asynchronous Evidence Handling**: Fast ingestion with queued processing during analysis
- **Hypothesis Evolution**: Confidence-based status transitions rather than deletion/replacement
- **Revival Mechanisms**: Dormant theories can return to consideration when new evidence emerges
- **Oscillation Detection**: Recognizes analytical ambiguity and preserves multiple interpretations
- **Investigation Management**: Complete tracking of reasoning sessions with persistent results
- **Simple Loop Counter Failsafe**: Multiple protection layers against infinite reasoning
- **RESTful Web API**: Production-ready HTTP service with OpenAPI documentation
- **Webhook Integration**: Built-in support for Salesforce, monitoring systems, and custom integrations

## Architecture

### Single SLM Advantage
**Why single SLM over multi-agent systems?** While agentic systems can overcome LLM limitations, they lose the unified perspective that makes human reasoning effective and consume significantly more tokens through inter-agent communication.

**Our approach:** Give a single SLM the same "unified view" that human minds naturally possess by presenting all hypotheses, evidence, and relationships simultaneously as structured, token-efficient tuples.

### Dual-Thread Design
- **Context Processing Thread**: Fast, non-blocking evidence ingestion (<10ms response)
- **Deep Thought Reasoning Thread**: Protected analytical reasoning on stable context snapshots

### Token-Optimized Data Structures
```python
@dataclass
class ContextItem:
    content: str                # Evidence or hypothesis content
    timestamp: datetime         # Creation time
    item_type: ItemType        # EVIDENCE | HYPOTHESIS | REVIVAL | PATTERN | SUMMARY
    status: Status             # ACTIVE | WEAKENED | DORMANT
    confidence: float          # 0.1 to 0.95 confidence level
    importance: float          # System-assessed importance
    source: str               # Evidence source identifier
    tags: List[str]           # Categorization tags
    access_count: int         # Number of times accessed
    reasoning_version: int    # Last reasoning cycle processed
```

### Compact Assertion Tuple Format
Evidence is presented to the SLM as structured tuples designed for minimal token usage:
```
(Enterprise Q4 sales exceeded targets by 30%, 14:23, conf:0.85, active, 5min_ago, imp:0.80, sales_system)
(SMB market requires different approach, 14:18, conf:0.70, active, 10min_ago, imp:0.90, llm_generation)
(Infrastructure scaling is adequate, 14:12, conf:0.40, weakened, 15min_ago, imp:0.70, llm_generation)
```

This format provides complete context while minimizing token consumption for cost-effective long-running business analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mossrake/async-reasoning-engine.git
cd async-reasoning-engine
```

2. Install dependencies:
```bash
pip install fastapi uvicorn dspy-ai python-multipart
```

3. Set up environment variables for Azure OpenAI (optimized for smaller models):
```bash
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT="your-deployment" # Recommended: gpt-4o-mini for cost efficiency
export AZURE_OPENAI_VERSION="2024-08-01-preview"
```

## Quick Start

### Basic Usage
```python
from reasoning import AsyncReasoningEngine

# Initialize with token budget optimized for SLM deployment
engine = AsyncReasoningEngine(max_context_tokens=4000)
engine.start()

# Start a new investigation
investigation_id = engine.clear_context("customer_analysis")

# Add evidence (automatically formatted as compact tuples)
engine.add_evidence("Sales metrics show 30% growth", "sales_system", confidence=0.85)
engine.add_evidence("Customer complaints increased 40%", "support_system", confidence=0.90)

# Query the reasoning state
response = engine.query_context_sync("What does the evidence suggest about business performance?")
print(response)

# Check status
status = engine.get_status_snapshot()
print(f"Hypotheses: {status['hypotheses']}, Evidence: {status['evidence_items']}")

# Retrieve investigation results
results = engine.get_investigation_results(investigation_id)
print(f"Investigation complete: {len(results['hypotheses'])} hypotheses generated")

engine.stop()
```

### Web Service API
Start the FastAPI web service:

**Development Mode:**
```bash
python webservice.py dev
```

**Production Mode:**
```bash
python webservice.py prod
```

Access the interactive API documentation at `http://localhost:8000/docs`

#### Core API Endpoints
- `POST /evidence` - Add evidence to the reasoning engine
- `POST /hypothesis` - Add a hypothesis for consideration
- `POST /context/clear` - Clear context and start new investigation
- `GET /investigations/{id}` - Retrieve investigation results
- `POST /query` - Query reasoning state with natural language
- `GET /status` - Get engine status and statistics
- `POST /force_stop` - Force stop reasoning cycles (human override)
- `POST /config/max_loops` - Configure maximum reasoning loops

#### Integration Endpoints
- `POST /webhooks/salesforce` - Salesforce event webhook
- `POST /webhooks/monitoring` - Monitoring system alerts
- `GET /health` - Detailed health check with performance metrics

#### Debug Endpoints
- `GET /debug/context` - View full reasoning context (development only)
- `POST /debug/toggle_full_context` - Toggle detailed reasoning display

### Example API Usage
```bash
# Start new investigation
curl -X POST "http://localhost:8000/context/clear" \
  -H "Content-Type: application/json" \
  -d '{"base_name": "q4_analysis"}'

# Add evidence (automatically converted to token-efficient format)
curl -X POST "http://localhost:8000/evidence" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Q4 sales exceeded targets by 30%",
    "source": "sales_system",
    "confidence": 0.85
  }'

# Query reasoning
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What evidence supports enterprise focus?"}'

# Get investigation results
curl -X GET "http://localhost:8000/investigations/q4_analysis_20250815_143022_001"
```

## Configuration

### Token Budget Management
```python
max_context_tokens: int = 4000        # SLM context window budget (optimized for cost efficiency)
compression_threshold: int = 3200     # Trigger context compression to maintain token limits
max_sterile_cycles: int = 3           # Reasoning termination threshold
max_reasoning_cycles: int = 10        # Hard reasoning limit
max_reasoning_loops: int = 10         # Simple loop counter failsafe
oscillation_detection_window: int = 6 # Pattern detection window
max_deep_thought_minutes: int = 5     # Deep thought timeout
assertion_batch_size: int = 5        # Evidence processing batch size
```

### Cost-Effective SLM Deployment
The engine is specifically designed for **Small Language Models** to minimize operational costs:
- **Compact tuple format** reduces token consumption by ~40% vs. verbose representations
- **Intelligent context compression** maintains reasoning quality within token budgets
- **Batch processing** optimizes API calls for cost efficiency
- **Configurable token limits** prevent cost overruns in production

### Failsafe Configuration
The engine includes multiple protection mechanisms:
1. **Simple Loop Counter**: Hard limit on reasoning loops per session
2. **Sterile Cycle Detection**: Stops when no changes occur for N cycles
3. **Oscillation Detection**: Recognizes cyclic patterns and preserves multiple interpretations
4. **Deep Thought Timeout**: Prevents indefinite reasoning sessions
5. **Human Override**: API endpoint for manual intervention

```python
# Configure maximum loops for cost control
engine.configure_max_loops(15)

# Force stop if reasoning costs exceed budget
engine.force_stop_reasoning("Token budget exceeded")
```

## How It Works

### Token-Efficient Reasoning Process
The engine uses **structured assertion tuples** to maximize reasoning within constrained token budgets:

1. **Compact Evidence Representation**: Each piece of evidence becomes a concise tuple
2. **Unified Context View**: All hypotheses and evidence visible simultaneously to SLM
3. **Progressive Reasoning**: Confidence updates happen gradually across reasoning cycles
4. **Intelligent Compression**: Low-importance items removed when approaching token limits

### Investigation Lifecycle
1. **Start Investigation**: `clear_context()` captures previous results and starts fresh
2. **Evidence Accumulation**: Add evidence through API or direct calls (auto-formatted as tuples)
3. **Deep Thought Processing**: SLM analyzes evidence and generates hypotheses within token budget
4. **Confidence Evolution**: Hypotheses strengthen or weaken based on supporting evidence
5. **Natural Completion**: Reasoning stops when stability is reached or limits hit
6. **Result Retrieval**: Access complete analysis through investigation ID

### Hypothesis Evolution
1. **Evidence Arrival**: New data triggers reasoning cycles
2. **Confidence Adjustment**: Hypotheses gain/lose confidence based on supporting/contradicting evidence
3. **Status Transitions**: Active → Weakened → Dormant based on confidence thresholds
4. **Revival Process**: Dormant hypotheses return to active status when supporting evidence emerges

### SLM-Driven Reasoning
- **Direct State Management**: SLM analyzes context and directly sets hypothesis confidence/status
- **JSON Decision Format**: Structured output eliminates keyword parsing complexity
- **Context-Only Analysis**: Pure Markov chain - each step only depends on current context window
- **Token-Aware Processing**: Reasoning adapts to available context space

### Deep Thought Mode
When evidence accumulates, the engine enters "deep thought":
- New evidence gets queued (not lost)
- SLM works with stable context snapshot
- Reasoning continues until stability or token limits reached
- Detailed progress reporting for each reasoning cycle
- Multiple termination conditions prevent cost overruns

### Oscillation Detection
The system recognizes when evidence supports contradictory theories equally:
- Detects 2-state and 3-state oscillation patterns
- Preserves multiple valid interpretations
- Prevents infinite reasoning loops (and associated costs)
- Reports oscillation patterns for human analysis

## Performance & Cost Efficiency

- **Evidence ingestion**: <10ms average response time
- **Context processing**: <50ms for categorization and queuing
- **Deep thought sessions**: 10-60 seconds for complete analysis (token-budget controlled)
- **Reasoning cycles**: 1-5 seconds per cycle within deep thought
- **API response times**: <100ms for non-reasoning operations
- **Investigation result capture**: <100ms for context clearing
- **Token efficiency**: ~40% reduction vs. verbose multi-agent approaches

## Monitoring and Observability

The engine provides comprehensive monitoring through the `/status` endpoint:

```json
{
  "engine_running": true,
  "total_items": 15,
  "hypotheses": 3,
  "evidence_items": 12,
  "deep_thought_mode": false,
  "reasoning_cycles_total": 7,
  "reasoning_loop_count": 0,
  "max_reasoning_loops": 10,
  "consecutive_sterile_cycles": 0,
  "reasoning_needed": false,
  "token_usage": {
    "current_context_tokens": 2850,
    "max_context_tokens": 4000,
    "compression_triggered": false,
    "utilization_percentage": 71.25
  },
  "stats": {
    "assertions_processed": 15,
    "reasoning_cycles": 7,
    "hypotheses_generated": 3,
    "revivals_detected": 1,
    "compressions_performed": 0,
    "context_changes": 8
  }
}
```

### Cost Monitoring
The `/health` endpoint provides detailed cost and efficiency assessment:
- Token usage tracking and budget alerts
- Cost-per-investigation metrics
- SLM efficiency indicators
- Compression performance statistics

### Progress Reporting
During deep thought sessions, the engine provides detailed cycle-by-cycle progress:
- Token consumption per reasoning cycle
- Confidence updates for each hypothesis
- Status changes (active/weakened/dormant transitions)
- Hypothesis generation and deletion tracking
- Revival detection
- Evidence impact assessment

## Use Cases

- **Business Intelligence**: Handling contradictory KPIs and metrics with cost-effective analysis
- **Security Analysis**: Managing conflicting threat indicators within operational budgets
- **Market Research**: Tracking competing market theories with minimal token overhead
- **Operational Monitoring**: Balancing performance vs. reliability signals efficiently
- **Strategic Planning**: Maintaining multiple scenario hypotheses cost-effectively
- **Customer Analysis**: Reconciling satisfaction vs. retention data
- **Financial Analysis**: Managing conflicting market signals with controlled costs

## Webhook Integrations

### Salesforce Integration
```bash
curl -X POST "http://localhost:8000/webhooks/salesforce" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "opportunity_closed_won",
    "data": {
      "amount": 50000,
      "account_type": "enterprise",
      "close_date": "2024-03-15"
    }
  }'
```

### Monitoring System Integration
```bash
curl -X POST "http://localhost:8000/webhooks/monitoring" \
  -H "Content-Type: application/json" \
  -d '{
    "alert_name": "High CPU Usage",
    "severity": "warning",
    "message": "CPU usage exceeded 80% for 5 minutes",
    "source": "production-web-01"
  }'
```

## Technical Innovation

### Why Single SLM vs Multi-Agent?
Our unified approach provides **cost-effective reasoning** that replicates human cognitive advantages:
- **Integrated Reasoning**: All evidence-hypothesis relationships visible in each cycle
- **Natural Hypothesis Interaction**: Competing theories compared directly
- **Seamless Revival Logic**: Dormant hypotheses remain contextually available
- **Unified Confidence Assessment**: Evidence evaluated against complete landscape
- **Token Efficiency**: ~60% fewer tokens than equivalent multi-agent systems
- **Cost Predictability**: Single model deployment with controlled resource usage

### Queue Architecture Innovation
Single-queue design with boundary operations:
- All operations (evidence, hypotheses, clears) use unified queue
- Context clearing acts as investigation boundaries
- Clean separation between reasoning sessions
- No race conditions between operation types
- Minimal token overhead for operation management

### Multiple Failsafe Layers
- **Simple Loop Counter**: Prevents runaway reasoning (and associated costs)
- **Sterile Cycle Detection**: Stops when no progress is made
- **Oscillation Detection**: Preserves valid contradictory interpretations
- **Deep Thought Timeout**: Time-based reasoning limits
- **Token Budget Enforcement**: Hard caps on context consumption
- **Human Override**: Manual intervention capability

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "webservice.py", "prod"]
```

### Environment Configuration
```bash
# Required
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # Recommended for cost efficiency

# Optional - Cost Control
AZURE_OPENAI_VERSION="2024-10-21"
MAX_REASONING_LOOPS=10
DEEP_THOUGHT_TIMEOUT_MINUTES=5
MAX_CONTEXT_TOKENS=4000  # Optimized for SLM deployment
```

### Scaling Considerations
- **Cost-Effective Scaling**: Single SLM per investigation domain
- **Resource Predictability**: Controlled token usage enables accurate cost forecasting
- **Horizontal Scaling**: Multiple service instances with sticky session routing
- **Budget Controls**: Configurable limits prevent cost overruns

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The underlying asynchronous reasoning framework and architectural design patterns implemented in this software are proprietary intellectual property of Mossrake Group, LLC.

## Contributing

Please read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for detailed technical background and implementation details.

### Development Setup
```bash
git clone https://github.com/mossrake/async-reasoning-engine.git
cd async-reasoning-engine
pip install -r requirements.txt
python webservice.py dev  # Start in development mode with auto-reload
```

## Documentation

- [Technical Overview] - Complete technical implementation details
- [Living in the Contradiction] - Conceptual framework and motivation

## Support

For issues and questions, please use the GitHub Issues tracker.

## Changelog

### Recent Updates
- **SLM Optimization**: Token-efficient tuple format for cost-effective deployment
- **Investigation Management**: Complete investigation lifecycle with persistent results
- **Simple Loop Counter Failsafe**: Additional protection against infinite reasoning and cost overruns
- **Web Service API**: Production-ready FastAPI service with comprehensive endpoints
- **Webhook Integration**: Built-in support for Salesforce, monitoring systems
- **Enhanced Monitoring**: Detailed health checks, progress reporting, and cost tracking
- **Multiple Failsafe Layers**: Comprehensive protection against reasoning failures and budget overruns
