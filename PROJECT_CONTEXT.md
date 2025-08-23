# Project Context: Asynchronous Reasoning Engine

## Vision and Purpose

The Asynchronous Reasoning Engine represents a fundamental shift in how AI systems handle contradictory information while maintaining **cost-effective operation through Small Language Model (SLM) deployment**. Rather than forcing immediate resolution between conflicting data signals, this system embraces the human cognitive ability to "live in the contradiction" until sufficient evidence accumulates to clarify the situation - all while minimizing token consumption for long-running reasoning in business applications.

### The Problem We're Solving

Business environments consistently generate noisy, contradictory data streams:
- Sales metrics suggest growth while customer satisfaction scores decline
- Security alerts indicate threats while operational metrics appear normal  
- Performance dashboards show optimal response times while users report slowness

Traditional reasoning systems struggle with this contradiction, either:
1. Forcing premature resolution between contradictory signals
2. Oscillating between different interpretations with each new data point
3. Discarding potentially valid alternative explanations to maintain consistency
4. Providing artificially certain conclusions that reverse with the next data point
5. Consuming many tokens through verbose multi-agent communication

### Our Approach: Token-Efficient Single SLM with Protected Reasoning

Instead of building complex multi-agent systems or deploying expensive large language models, we use a **single Small Language Model (SLM)** managing a rewritable context window filled with structured assertion tuples designed for minimal token consumption.

## Core Technical Innovations

### 1. Token-Optimized Structured Assertion Tuples
Every piece of evidence and hypothesis becomes a **compact timestamped tuple** designed for maximum information density:
```
(content, timestamp, conf:X.XX, status, age_minutes, imp:X.XX, source)
```

This **minimizes token consumption by an estimated ~40%** compared to verbose natural language representations while providing the SLM with complete structured context containing all hypotheses, evidence, and relationships simultaneously.

### 2. Deep Thought Processing with Budget Controls
Protected reasoning sessions where the SLM analyzes stable context snapshots without interference from new evidence. This mirrors how human analysts need uninterrupted thinking time while maintaining strict token budget enforcement for cost-effective deployment.

### 3. Asynchronous Evidence Handling
Fast ingestion with queued processing during analysis prevents reactive oscillation while ensuring no evidence is lost and no tokens are wasted on premature context switches.

### 4. Hypothesis Evolution with Minimal Overhead
Confidence-based status transitions (active/weakened/dormant) rather than deletion/replacement, allowing theories to persist and be revived when circumstances change - all represented within the same token-efficient tuple structure.

### 5. Investigation Lifecycle Management
Complete investigation tracking with persistent results storage:
- Unique investigation IDs for each reasoning session
- Context clearing operations that capture results before reset
- Investigation results retrieval for historical analysis
- Automatic investigation sequencing and timestamping

### 6. Simple Loop Counter Failsafe with Cost Protection
Additional protection against infinite reasoning **and associated cost overruns**:
- Hard maximum loop count per reasoning session (default: 10)
- Simple counter that resets with each reasoning session
- Failsafe triggers before other termination conditions
- Configurable limits for different deployment scenarios and budget constraints

## Architectural Philosophy

### Single SLM vs Multi-Agent Systems

**Why not multi-agent?** While agentic systems can overcome LLM limitations (sequential processing, context constraints), they lose the unified perspective that makes human reasoning effective and consume 3-5x more tokens through inter-agent communication overhead.

**Our approach:** Give a single SLM the same "unified view" that human minds naturally possess by presenting all hypotheses, evidence, and relationships simultaneously as structured, token-efficient context that maximizes reasoning capability within constrained budgets.

**Benefits:**
- Integrated reasoning across all evidence-hypothesis relationships
- Natural hypothesis interaction and comparison
- Seamless revival logic for dormant theories
- Unified confidence assessment against complete context
- Fewer tokens than equivalent multi-agent approaches

### Thread Architecture

**Context Processing Thread:**
- Fast, non-blocking evidence ingestion (<10ms)
- Validates and categorizes incoming assertions
- Maintains responsiveness during reasoning sessions
- Processes single operation queue with clear boundary logic
- Handles context clearing operations asynchronously

**Deep Thought Reasoning Thread:**
- Monitors context version changes
- Executes protected analytical reasoning within token budgets
- Manages hypothesis confidence evolution
- Detects completion and oscillation patterns
- Simple loop counter failsafe protection
- Intelligent context compression when approaching token limits

### Queue Architecture
Unified single-queue design optimized for token efficiency:
- All operations (evidence, hypotheses, clears, queries) use single queue
- Clear operations act as boundaries for reasoning sessions
- Operations before clear are processed in batches
- Clear operations capture results and reset context

## Implementation Highlights

### Token Budget Management System
```python
context_version: int          # Incremented on each context change
last_reasoned_version: int    # Last version processed by reasoning
context_change_event          # Thread synchronization primitive
reasoning_loop_count: int     # Simple counter for current session
max_reasoning_loops: int      # Configurable failsafe limit
current_token_count: int      # Real-time token usage tracking
compression_threshold: int    # Trigger intelligent context compression
```

### Hash-Based Termination with Cost Control
Deep thought sessions continue until analytical stability, detected through context hashing. When consecutive reasoning cycles produce identical context hashes, stable conclusions have been reached without exceeding token budgets.

### Oscillation Detection
Recognizes 2-state and 3-state oscillation patterns in reasoning cycles. When evidence genuinely supports contradictory theories equally, the system preserves both interpretations rather than cycling infinitely.

### Hypothesis Lifecycle Management
- **Active** (conf > 0.5): Primary working theories
- **Weakened** (0.3 < conf ≤ 0.5): Secondary considerations  
- **Dormant** (conf ≤ 0.3): Shelved but recoverable theories

Revival mechanism: Dormant hypotheses first return to active status, then gain confidence in subsequent cycles.

### Investigation Management
Complete investigation lifecycle with cost tracking:
```python
investigation_id = engine.clear_context("investigation_name")
# Token-efficient reasoning happens...
results = engine.get_investigation_results(investigation_id)
# Results include token usage and cost metrics
```

Results include:
- Final hypothesis states and confidence levels
- Evidence summary and source breakdown
- Reasoning cycle statistics
- Timestamps and processing metrics

## Performance Characteristics

### Timing Benchmarks
- Evidence ingestion: <10ms average response
- Context processing: <50ms for categorization and queuing
- Deep thought sessions: 10-60 seconds for complete analysis (budget-controlled)
- Reasoning cycles: 1-5 seconds per cycle within deep thought
- Context clearing: <100ms for result capture


### Memory and Cost Management
- **Context compression**: Automatic removal of low-importance items when approaching token limits
- **Hash history**: Rolling window for oscillation detection
- **Queue management**: Bounded queues with overflow handling
- **Investigation results storage**: SQLite stub implementation
- **Cost tracking**: Per-investigation token usage and pricing
- **Budget enforcement**: Hard caps prevent cost overruns

### Scaling Considerations
- **Configurable token budgets** with intelligent compression
- **Hard caps** on reasoning duration and cycle count
- **Simple loop counter** provides additional cost protection
- **RLock protection** for thread-safe context modifications
- **Predictable resource usage** enables accurate cost forecasting

## Business Value Proposition

### Cost-Effective Uncertainty Handling
When business data genuinely supports multiple interpretations, the system acknowledges this rather than forcing artificial certainty *while maintaining operational costs that are much lower than multi-agent alternatives.

### Stable Analysis with Budget Predictability
Conclusions evolve gradually rather than whipsawing between interpretations with each new data point, all within controlled token budgets that enable accurate cost forecasting.

### Complete Context within Token Constraints
Decision-makers see the full landscape of possibilities, including theories that were considered and set aside.

### Revival Capability
Previously dismissed explanations can return to consideration when new evidence makes them relevant again.

### Investigation Continuity with Cost Attribution
Complete tracking of reasoning sessions with retrievable historical results enables:
- Comparative analysis across different evidence sets
- Historical decision audit trails
- Pattern recognition across multiple investigations
- Persistent knowledge base development

## Development Philosophy

### Code Organization
- **reasoning.py**: Core engine implementation with dual-thread architecture and token management
- **webservice.py**: FastAPI web service providing HTTP endpoints with cost monitoring
- Separation of concerns between fast assertion processing and deep reasoning
- Complete web API with investigation management and cost tracking endpoints

### Cost-Conscious Error Handling Strategy
- LLM failure recovery with graceful degradation
- Context corruption protection with hash validation
- Thread synchronization with deadlock prevention
- Timeout handling for all operations
- Multiple failsafe layers (hash-based, sterile cycles, loop counter)

### Monitoring and Observability
- Real-time status reporting (engine state, hypothesis counts)
- Performance metrics (processing times, cycle efficiency)
- Reasoning audit trails (complete hypothesis evolution history)
- Oscillation alerts (automatic detection of analytical ambiguity)
- Deep thought progress reporting with detailed cycle analysis
- Investigation result persistence and retrieval

## Future Enhancements

### Planned Features
- **Hypothesis Merging**: Automatic combination of similar theories with token optimization
- **Evidence Source Weighting**: Dynamic confidence adjustment based on source reliability
- **Multi-Domain Reasoning**: Parallel reasoning across different knowledge domains
- **Temporal Reasoning**: Time-based evidence decay and trend analysis
- **SQLite Integration**: Replace stub investigation storage with full database
- **Investigation Analytics**: Pattern analysis across historical investigations
- **Advanced Token Optimization**: Dynamic tuple compression and context pruning
- **Cost management integrations** with cloud billing and budget systems

### Integration Patterns
- Message queue integration (Kafka/RabbitMQ) for high-volume evidence streams
- Database persistence for context state durability and recovery
- Microservice deployment with containerized reasoning services
- Real-time dashboards for live reasoning state visualization
- Webhook integrations for Salesforce, monitoring systems
- RESTful API with OpenAPI documentation

## Technical Dependencies

### Core Dependencies
- **DSPy**: LLM integration and prompt management (Azure OpenAI compatible)
- **FastAPI**: Web service framework with auto-generated documentation
- **Azure OpenAI**: SLM inference backend (recommended: gpt-4o-mini for cost efficiency)
- **Python 3.8+**: Runtime environment
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Request/response validation and serialization

### Environment Variables
```bash
AZURE_OPENAI_KEY          # Azure OpenAI API key
AZURE_OPENAI_ENDPOINT     # Azure OpenAI service endpoint
AZURE_OPENAI_DEPLOYMENT   # Azure OpenAI deployment name (recommended: gpt-4o-mini)
AZURE_OPENAI_VERSION      # API version (used: 2024-08-01-preview)
MAX_CONTEXT_TOKENS        # Token budget limit (default: 4000)
COMPRESSION_THRESHOLD     # Context compression trigger (default: 3200)
```

## Design Patterns and Best Practices

### Token-Efficient Context Management
- **Structured tuple format** maximizes information density
- **Intelligent compression** maintains reasoning quality within budgets
- **Version tracking** for reasoning cycle coordination
- **Investigation boundary management** with cost-aware result capture

### Thread Safety
- RLock protection for all context modifications
- Queue-based communication between threads
- Event-based synchronization for reasoning triggers
- Single queue design eliminates race conditions

### SLM Integration
- **Structured prompts** optimized for token efficiency
- **Error handling and retry logic** for SLM failures
- **Token budget management** with intelligent compression
- **Azure OpenAI compatible** DSPy implementation optimized for smaller models

### Anti-Patterns Avoided
- **Token Waste Prevention**: Efficient tuple representation vs. verbose natural language
- **Rumination Prevention**: Explicit instructions to SLM to avoid re-analyzing existing items
- **Oscillation Mitigation**: Pattern detection and explicit termination
- **Context Pollution**: Careful management of what gets added to reasoning context
- **Infinite Loops**: Multiple failsafe mechanisms including cost-aware loop counter
- **Budget Overruns**: Hard token limits with compression and termination

## Web Service Architecture

### API Design Principles
- RESTful endpoint design with clear resource boundaries
- Comprehensive request/response validation
- Detailed error handling with appropriate HTTP status codes
- OpenAPI documentation with interactive testing interface

### Key Endpoints
```python
POST /evidence           # Add evidence to reasoning engine (auto-formatted as tuples)
POST /hypothesis         # Add hypothesis for consideration
POST /context/clear      # Clear context and start new investigation
GET  /investigations/{id} # Retrieve investigation results with cost metrics
POST /query             # Natural language query interface
GET  /status            # Engine status, performance metrics, and token usage
POST /force_stop        # Human override for stuck reasoning or budget overruns
POST /config/max_loops  # Configure failsafe parameters
GET  /cost/analysis     # Cost analysis and token usage analytics
```

### Webhook Support
- Salesforce integration endpoint
- Monitoring system alert processing
- Extensible webhook framework for additional integrations

### Debug and Development
- Full context inspection endpoints
- Toggle full context display during reasoning
- Health check with detailed engine status
- Development vs production mode configuration

## Contribution Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Comprehensive docstrings for all public methods
- Type hints for all function parameters and return values
- Pydantic models for all API request/response schemas

### Testing Philosophy
- Unit tests for core reasoning logic
- Integration tests for API endpoints
- Performance benchmarks for timing-critical operations
- API endpoint testing with FastAPI test client

### Documentation Standards
- Technical decisions documented in code comments
- Architecture decisions captured in design documents
- API documentation auto-generated from OpenAPI schemas

## Intellectual Property Notes

The code is released under GNU Affero General Public License v3+, allowing redistribution and modification. However, the underlying asynchronous reasoning framework, token optimization techniques, and architectural design patterns are proprietary intellectual property of Mossrake Group, LLC.

This dual licensing approach allows open source usage while protecting the core innovations that make this system uniquely cost-effective for business deployment.
