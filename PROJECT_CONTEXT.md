# Project Context: Asynchronous Reasoning Engine

## Vision and Purpose

The Asynchronous Reasoning Engine represents a fundamental shift in how AI systems handle contradictory information. Rather than forcing immediate resolution between conflicting data signals, this system embraces the human cognitive ability to "live in the contradiction" until sufficient evidence accumulates to clarify the situation.

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

### Our Approach: Unified Context with Protected Reasoning

Instead of building complex multi-agent systems or forced belief revision frameworks, we use a single language model managing a rewritable context window filled with structured assertion tuples.

## Core Technical Innovations

### 1. Structured Assertion Tuples
Every piece of evidence and hypothesis becomes a timestamped tuple:
```
(content, timestamp, conf:X.XX, status, source, metadata)
```

This provides the LLM with complete structured context containing all hypotheses, evidence, and relationships simultaneously.

### 2. Deep Thought Processing
Protected reasoning sessions where the LLM analyzes stable context snapshots without interference from new evidence. This mirrors how human analysts need uninterrupted thinking time.

### 3. Asynchronous Evidence Handling
Fast ingestion with queued processing during analysis prevents reactive oscillation while ensuring no evidence is lost.

### 4. Hypothesis Evolution
Confidence-based status transitions (active/weakened/dormant) rather than deletion/replacement, allowing theories to persist and be revived when circumstances change.

### 5. Investigation Lifecycle Management
**NEW**: Complete investigation tracking with persistent results storage:
- Unique investigation IDs for each reasoning session
- Context clearing operations that capture results before reset
- Investigation results retrieval for historical analysis
- Automatic investigation sequencing and timestamping

### 6. Simple Loop Counter Failsafe
**NEW**: Additional protection against infinite reasoning:
- Hard maximum loop count per reasoning session (default: 10)
- Simple counter that resets with each new reasoning session
- Failsafe triggers before other termination conditions
- Configurable limits for different deployment scenarios

## Architectural Philosophy

### Single LLM vs Multi-Agent Systems

**Why not multi-agent?** While agentic systems can overcome LLM limitations (sequential processing, context constraints), they lose the unified perspective that makes human reasoning effective.

**Our breakthrough:** Give a single LLM the same "unified view" that human minds naturally possess by presenting all hypotheses, evidence, and relationships simultaneously as structured context.

**Benefits:**
- Integrated reasoning across all evidence-hypothesis relationships
- Natural hypothesis interaction and comparison
- Seamless revival logic for dormant theories
- Unified confidence assessment against complete context

### Thread Architecture

**Context Processing Thread:**
- Fast, non-blocking evidence ingestion (<10ms)
- Validates and categorizes incoming assertions
- Maintains responsiveness during reasoning sessions
- **NEW**: Processes single operation queue with clear boundary logic
- **NEW**: Handles context clearing operations asynchronously

**Deep Thought Reasoning Thread:**
- Monitors context version changes
- Executes protected analytical reasoning
- Manages hypothesis confidence evolution
- Detects completion and oscillation patterns
- **NEW**: Simple loop counter failsafe protection

### Queue Architecture
**NEW**: Unified single-queue design replaces separate queues:
- All operations (evidence, hypotheses, clears, queries) use single queue
- Clear operations act as boundaries for reasoning sessions
- Operations before clear are processed in batches
- Clear operations capture results and reset context

## Implementation Highlights

### Context Versioning System
```python
context_version: int          # Incremented on each context change
last_reasoned_version: int    # Last version processed by reasoning
context_change_event          # Thread synchronization primitive
reasoning_loop_count: int     # NEW: Simple counter for current session
max_reasoning_loops: int      # NEW: Configurable failsafe limit
```

### Hash-Based Termination
Deep thought sessions continue until analytical stability, detected through context hashing. When consecutive reasoning cycles produce identical context hashes, stable conclusions have been reached.

### Oscillation Detection
Recognizes 2-state and 3-state oscillation patterns in reasoning cycles. When evidence genuinely supports contradictory theories equally, the system preserves both interpretations rather than cycling infinitely.

### Hypothesis Lifecycle Management
- **Active** (conf > 0.5): Primary working theories
- **Weakened** (0.3 < conf ≤ 0.5): Secondary considerations  
- **Dormant** (conf ≤ 0.3): Shelved but recoverable theories

Revival mechanism: Dormant hypotheses first return to active status, then gain confidence in subsequent cycles.

### Investigation Management
**NEW**: Complete investigation lifecycle:
```python
investigation_id = engine.clear_context("investigation_name")
# Reasoning happens...
results = engine.get_investigation_results(investigation_id)
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
- Deep thought sessions: 10-60 seconds for complete analysis
- Reasoning cycles: 1-5 seconds per cycle within deep thought
- **NEW**: Context clearing: <100ms for result capture

### Memory Management
- Context compression: Automatic removal of low-importance items
- Hash history: Rolling window for oscillation detection
- Queue management: Bounded queues with overflow handling
- **NEW**: Investigation results storage (SQLite stub implementation)

### Scaling Considerations
- Configurable token budgets with intelligent compression
- Hard caps on reasoning duration and cycle count
- **NEW**: Simple loop counter provides additional failsafe protection
- RLock protection for thread-safe context modifications

## Business Value Proposition

### Realistic Uncertainty Handling
When business data genuinely supports multiple interpretations, the system acknowledges this rather than forcing artificial certainty.

### Stable Analysis
Conclusions evolve gradually rather than whipsawing between interpretations with each new data point.

### Complete Context
Decision-makers see the full landscape of possibilities, including theories that were considered and set aside.

### Revival Capability
Previously dismissed explanations can return to consideration when new evidence makes them relevant again.

### **NEW**: Investigation Continuity
Complete tracking of reasoning sessions with retrievable historical results enables:
- Comparative analysis across different evidence sets
- Historical decision audit trails
- Pattern recognition across multiple investigations
- Persistent knowledge base development

## Development Philosophy

### Code Organization
- **reasoning.py**: Core engine implementation with dual-thread architecture
- **webservice.py**: FastAPI web service providing HTTP endpoints
- Separation of concerns between fast assertion processing and deep reasoning
- **NEW**: Complete web API with investigation management endpoints

### Error Handling Strategy
- LLM failure recovery with graceful degradation
- Context corruption protection with hash validation
- Thread synchronization with deadlock prevention
- Timeout handling for all operations
- **NEW**: Multiple failsafe layers (hash-based, sterile cycles, loop counter)

### Monitoring and Observability
- Real-time status reporting (engine state, hypothesis counts)
- Performance metrics (processing times, cycle efficiency)
- Reasoning audit trails (complete hypothesis evolution history)
- Oscillation alerts (automatic detection of analytical ambiguity)
- **NEW**: Deep thought progress reporting with detailed cycle analysis
- **NEW**: Investigation result persistence and retrieval

## Future Enhancements

### Planned Features
- **Hypothesis Merging**: Automatic combination of similar theories
- **Evidence Source Weighting**: Dynamic confidence adjustment based on source reliability
- **Multi-Domain Reasoning**: Parallel reasoning across different knowledge domains
- **Temporal Reasoning**: Time-based evidence decay and trend analysis
- **SQLite Integration**: Replace stub investigation storage with full database
- **Investigation Analytics**: Pattern analysis across historical investigations

### Integration Patterns
- Message queue integration (Kafka/RabbitMQ) for high-volume evidence streams
- Database persistence for context state durability and recovery
- Microservice deployment with containerized reasoning services
- Real-time dashboards for live reasoning state visualization
- **NEW**: Webhook integrations for Salesforce, monitoring systems
- **NEW**: RESTful API with OpenAPI documentation

## Technical Dependencies

### Core Dependencies
- **DSPy**: LLM integration and prompt management (Azure OpenAI compatible)
- **FastAPI**: Web service framework with auto-generated documentation
- **Azure OpenAI**: LLM inference backend
- **Python 3.8+**: Runtime environment
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Request/response validation and serialization

### Environment Variables
```bash
AZURE_OPENAI_KEY          # Azure OpenAI API key
AZURE_OPENAI_ENDPOINT     # Azure OpenAI service endpoint
AZURE_OPENAI_DEPLOYMENT   # Azure OpenAI deployment name
AZURE_OPENAI_VERSION      # API version (default: 2024-02-15-preview)
```

## Design Patterns and Best Practices

### Context Management
- Immutable assertion content with mutable metadata
- Structured tuple format for LLM consumption
- Version tracking for reasoning cycle coordination
- **NEW**: Investigation boundary management with result capture

### Thread Safety
- RLock protection for all context modifications
- Queue-based communication between threads
- Event-based synchronization for reasoning triggers
- **NEW**: Single queue design eliminates race conditions

### LLM Integration
- Structured prompts for different reasoning tasks
- Error handling and retry logic for LLM failures
- Token budget management with intelligent compression
- **NEW**: Azure OpenAI compatible DSPy implementation

### Anti-Patterns Avoided
- **Rumination Prevention**: Explicit instructions to LLM to avoid re-analyzing existing items
- **Oscillation Mitigation**: Pattern detection and explicit termination
- **Context Pollution**: Careful management of what gets added to reasoning context
- **Infinite Loops**: Multiple failsafe mechanisms including simple loop counter

## Web Service Architecture

### API Design Principles
- RESTful endpoint design with clear resource boundaries
- Comprehensive request/response validation
- Detailed error handling with appropriate HTTP status codes
- OpenAPI documentation with interactive testing interface

### Key Endpoints
```python
POST /evidence           # Add evidence to reasoning engine
POST /hypothesis         # Add hypothesis for consideration
POST /context/clear      # Clear context and start new investigation
GET  /investigations/{id} # Retrieve investigation results
POST /query             # Natural language query interface
GET  /status            # Engine status and performance metrics
POST /force_stop        # Human override for stuck reasoning
POST /config/max_loops  # Configure failsafe parameters
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
- **NEW**: Pydantic models for all API request/response schemas

### Testing Philosophy
- Unit tests for core reasoning logic
- Integration tests for API endpoints
- Performance benchmarks for timing-critical operations
- **NEW**: API endpoint testing with FastAPI test client

### Documentation Standards
- Technical decisions documented in code comments
- Architecture decisions captured in design documents
- **NEW**: API documentation auto-generated from OpenAPI schemas

## Intellectual Property Notes

The code is released under GNU Affero General Public License v3+, allowing redistribution and modification. However, the underlying asynchronous reasoning framework and architectural design patterns are proprietary intellectual property of Mossrake Group, LLC.

This dual licensing approach allows open source usage while protecting the core innovations that make this system unique.
