# Asynchronous Reasoning Engine

A reasoning engine for decision support that tracks multiple contradictory hypotheses over noisy/asynchronous business data, reviving dormant theories with new evidence.

## Overview

The Asynchronous Reasoning Engine replicates human cognitive flexibility by maintaining multiple competing explanations simultaneously without forcing immediate resolution. Unlike traditional AI systems that require consistency maintenance, this engine embraces contradictory hypotheses as a feature rather than a logical error.

### Key Innovation: Single LLM with Structured Context

Instead of complex multi-agent systems, we use a single language model managing a rewritable context window filled with structured assertion tuples. Each piece of evidence and hypothesis becomes a timestamped tuple containing content, confidence level, status, source, and metadata.

## Features

- **Living with Contradictions**: Maintains multiple competing hypotheses simultaneously
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

### Dual-Thread Design
- **Context Processing Thread**: Fast, non-blocking evidence ingestion (<10ms response)
- **Deep Thought Reasoning Thread**: Protected analytical reasoning on stable context snapshots

### Data Structures
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

### Assertion Tuple Format
Evidence is presented to the LLM as structured tuples:
```
(Enterprise Q4 sales exceeded targets by 30%, 14:23, conf:0.85, active, 5min_ago, imp:0.80, sales_system)
(SMB market requires different approach, 14:18, conf:0.70, active, 10min_ago, imp:0.90, llm_generation)
(Infrastructure scaling is adequate, 14:12, conf:0.40, weakened, 15min_ago, imp:0.70, llm_generation)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/async-reasoning-engine.git
cd async-reasoning-engine
```

2. Install dependencies:
```bash
pip install fastapi uvicorn dspy-ai python-multipart
```

3. Set up environment variables for Azure OpenAI:
```bash
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT="your-deployment" (used: gpt-4o-mini)
export AZURE_OPENAI_VERSION="2024-08-01-preview"
```

## Quick Start

### Basic Usage
```python
from reasoning import AsyncReasoningEngine

# Initialize and start the engine
engine = AsyncReasoningEngine(max_context_tokens=4000)
engine.start()

# Start a new investigation
investigation_id = engine.clear_context("customer_analysis")

# Add evidence
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

# Add evidence
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

### Engine Parameters
```python
max_context_tokens: int = 4000        # LLM context window budget
compression_threshold: int = 3200     # Trigger context compression
max_sterile_cycles: int = 3           # Reasoning termination threshold
max_reasoning_cycles: int = 10        # Hard reasoning limit
max_reasoning_loops: int = 10         # Simple loop counter failsafe
oscillation_detection_window: int = 6 # Pattern detection window
max_deep_thought_minutes: int = 5     # Deep thought timeout
assertion_batch_size: int = 5        # Evidence processing batch size
```

### Failsafe Configuration
The engine includes multiple protection mechanisms:
1. **Simple Loop Counter**: Hard limit on reasoning loops per session
2. **Sterile Cycle Detection**: Stops when no changes occur for N cycles
3. **Oscillation Detection**: Recognizes cyclic patterns and preserves multiple interpretations
4. **Deep Thought Timeout**: Prevents indefinite reasoning sessions
5. **Human Override**: API endpoint for manual intervention

```python
# Configure maximum loops
engine.configure_max_loops(15)

# Force stop if reasoning gets stuck
engine.force_stop_reasoning("Analysis taking too long")
```

## How It Works

### Investigation Lifecycle
1. **Start Investigation**: `clear_context()` captures previous results and starts fresh
2. **Evidence Accumulation**: Add evidence through API or direct calls
3. **Deep Thought Processing**: Engine analyzes evidence and generates hypotheses
4. **Confidence Evolution**: Hypotheses strengthen or weaken based on supporting evidence
5. **Natural Completion**: Reasoning stops when stability is reached or limits hit
6. **Result Retrieval**: Access complete analysis through investigation ID

### Hypothesis Evolution
1. **Evidence Arrival**: New data triggers reasoning cycles
2. **Confidence Adjustment**: Hypotheses gain/lose confidence based on supporting/contradicting evidence
3. **Status Transitions**: Active → Weakened → Dormant based on confidence thresholds
4. **Revival Process**: Dormant hypotheses return to active status when supporting evidence emerges

### LLM-Driven Reasoning
- **Direct State Management**: LLM analyzes context and directly sets hypothesis confidence/status
- **JSON Decision Format**: Structured output eliminates keyword parsing complexity
- **Context-Only Analysis**: Pure Markov chain - each step only depends on current context window

### Deep Thought Mode
When evidence accumulates, the engine enters "deep thought":
- New evidence gets queued (not lost)
- LLM works with stable context snapshot
- Reasoning continues until stability or limits reached
- Detailed progress reporting for each reasoning cycle
- Multiple termination conditions prevent infinite loops

### Oscillation Detection
The system recognizes when evidence supports contradictory theories equally:
- Detects 2-state and 3-state oscillation patterns
- Preserves multiple valid interpretations
- Prevents infinite reasoning loops
- Reports oscillation patterns for human analysis

## Performance

- **Evidence ingestion**: <10ms average response time
- **Context processing**: <50ms for categorization and queuing
- **Deep thought sessions**: 10-60 seconds for complete analysis
- **Reasoning cycles**: 1-5 seconds per cycle within deep thought
- **API response times**: <100ms for non-reasoning operations
- **Investigation result capture**: <100ms for context clearing

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

### Health Monitoring
The `/health` endpoint provides detailed health assessment:
- Engine operational status
- Deep thought duration warnings
- High reasoning loop usage alerts
- Performance degradation detection

### Progress Reporting
During deep thought sessions, the engine provides detailed cycle-by-cycle progress:
- Confidence updates for each hypothesis
- Status changes (active/weakened/dormant transitions)
- Hypothesis generation and deletion tracking
- Revival detection
- Evidence impact assessment

## Use Cases

- **Business Intelligence**: Handling contradictory KPIs and metrics
- **Security Analysis**: Managing conflicting threat indicators
- **Market Research**: Tracking competing market theories
- **Operational Monitoring**: Balancing performance vs. reliability signals
- **Strategic Planning**: Maintaining multiple scenario hypotheses
- **Customer Analysis**: Reconciling satisfaction vs. retention data
- **Financial Analysis**: Managing conflicting market signals

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

### Why Single LLM vs Multi-Agent?
Our unified approach replicates human cognitive advantages:
- **Integrated Reasoning**: All evidence-hypothesis relationships visible in each cycle
- **Natural Hypothesis Interaction**: Competing theories compared directly
- **Seamless Revival Logic**: Dormant hypotheses remain contextually available
- **Unified Confidence Assessment**: Evidence evaluated against complete landscape

### Queue Architecture Innovation
Single-queue design with boundary operations:
- All operations (evidence, hypotheses, clears) use unified queue
- Context clearing acts as investigation boundaries
- Clean separation between reasoning sessions
- No race conditions between operation types

### Multiple Failsafe Layers
- **Simple Loop Counter**: Prevents runaway reasoning (primary failsafe)
- **Sterile Cycle Detection**: Stops when no progress is made
- **Oscillation Detection**: Preserves valid contradictory interpretations
- **Deep Thought Timeout**: Time-based reasoning limits
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
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Optional
AZURE_OPENAI_VERSION="2024-10-21"
MAX_REASONING_LOOPS=10
DEEP_THOUGHT_TIMEOUT_MINUTES=5
```

### Scaling Considerations
- Stateful service (reasoning engine maintains context)
- Single instance per investigation domain
- Horizontal scaling through multiple service instances
- Load balancing requires sticky sessions or investigation routing

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The underlying asynchronous reasoning framework and architectural design patterns implemented in this software are proprietary intellectual property of Mossrake Group, LLC.

## Contributing

Please read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for detailed technical background and implementation details.

### Development Setup
```bash
git clone https://github.com/yourusername/async-reasoning-engine.git
cd async-reasoning-engine
pip install -r requirements.txt
python webservice.py dev  # Start in development mode with auto-reload
```

## Documentation -- see docs/

- [Technical Overview] - Complete technical implementation details
- [Living in the Contradiction] - Conceptual framework and motivation

## Support

For issues and questions, please use the GitHub Issues tracker.

## Changelog

### Recent Updates
- **Investigation Management**: Complete investigation lifecycle with persistent results
- **Simple Loop Counter Failsafe**: Additional protection against infinite reasoning
- **Web Service API**: Production-ready FastAPI service with comprehensive endpoints
- **Webhook Integration**: Built-in support for Salesforce and monitoring systems
- **Enhanced Monitoring**: Detailed health checks and progress reporting
- **Multiple Failsafe Layers**: Comprehensive protection against reasoning failures
