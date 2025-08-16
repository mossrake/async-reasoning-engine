"""
Copyright (c) 2025 Mossrake Group, LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

The underlying asynchronous reasoning framework and architectural design patterns
implemented in this software are proprietary intellectual property of Mossrake Group, LLC.
"""

"""
Production FastAPI Web Service for Reasoning Engine
Provides HTTP API endpoints for external systems to interact with the reasoning engine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import signal
import logging
from datetime import datetime

# Import the reasoning engine from reasoning.py
from reasoning import AsyncReasoningEngine

# Global reasoning engine instance
reasoning_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global reasoning_engine
    
    # Startup
    try:
        reasoning_engine = AsyncReasoningEngine(max_context_tokens=5000)
        reasoning_engine.start()
        
        logging.info("Reasoning engine started successfully")
        print("Reasoning Engine API started")
        print("   Endpoints available at /docs")
        print("   Reasoning engine: running")
        
        yield  # Application runs here
        
    except Exception as e:
        logging.error(f"Failed to start reasoning engine: {e}")
        raise RuntimeError(f"Startup failed: {e}")
    
    finally:
        # Shutdown
        if reasoning_engine:
            reasoning_engine.stop()
            logging.info("Reasoning engine stopped")
            print("Reasoning engine stopped")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Reasoning Engine API",
    description="API for adding evidence and querying reasoning state",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class EvidenceRequest(BaseModel):
    content: str = Field(..., description="Evidence content/description")
    source: str = Field(..., description="Source system or identifier")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")

class HypothesisRequest(BaseModel):
    content: str = Field(..., description="Hypothesis statement")
    confidence: float = Field(0.6, ge=0.0, le=1.0, description="Initial confidence score")
    source: str = Field("external_api", description="Source of hypothesis")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    timeout: float = Field(30.0, gt=0, description="Query timeout in seconds")

class ClearContextRequest(BaseModel):
    base_name: str = Field("investigation", description="Base name for investigation ID")

class ConfigurationRequest(BaseModel):
    max_loops: int = Field(..., ge=1, le=100, description="Maximum reasoning loops per session")

# NEW: Response models for missing endpoints
class ClearContextResponse(BaseModel):
    status: str
    investigation_id: str
    message: str

class InvestigationResults(BaseModel):
    investigation_id: str
    status: str
    timestamp: Optional[str]
    total_items: int
    evidence_count: int
    hypotheses_count: int
    reasoning_cycles: int
    hypotheses: List[Dict[str, Any]]
    evidence_summary: List[Dict[str, Any]]
    stats: Dict[str, Any]
    stub: bool = False

class EvidenceResponse(BaseModel):
    status: str
    operation_id: str
    message: str

class HypothesisResponse(BaseModel):
    status: str
    operation_id: str
    message: str

class QueryResponse(BaseModel):
    response: str
    processing_time_seconds: float

class StatusResponse(BaseModel):
    engine_running: bool
    total_items: int
    hypotheses: int
    evidence_items: int
    deep_thought_mode: bool
    deep_thought_duration_seconds: Optional[float]
    reasoning_cycles_total: int
    reasoning_loop_count: int
    max_reasoning_loops: int
    consecutive_sterile_cycles: int
    reasoning_needed: bool
    stats: Dict[str, Any]

class ForceStopRequest(BaseModel):
    reason: str = Field("API request", description="Reason for stopping reasoning")

# API Endpoints
@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "service": "Reasoning Engine API",
        "status": "running" if reasoning_engine and reasoning_engine.running else "stopped",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/evidence", response_model=EvidenceResponse, summary="Add Evidence")
async def add_evidence(evidence: EvidenceRequest):
    """
    Add evidence to the reasoning engine.
    
    The evidence will be processed asynchronously by the reasoning engine.
    Returns immediately with an operation ID for tracking.
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        start_time = datetime.now()
        operation_id = reasoning_engine.add_evidence(
            content=evidence.content,
            source=evidence.source,
            confidence=evidence.confidence
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logging.info(f"Evidence added: {evidence.content[:50]}... from {evidence.source}")
        
        return EvidenceResponse(
            status="queued",
            operation_id=operation_id,
            message=f"Evidence queued for processing in {processing_time*1000:.1f}ms"
        )
        
    except Exception as e:
        logging.error(f"Failed to add evidence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add evidence: {str(e)}")

@app.post("/hypothesis", response_model=HypothesisResponse, summary="Add Hypothesis") 
async def add_hypothesis(hypothesis: HypothesisRequest):
    """
    Add a hypothesis to the reasoning engine.
    
    Hypotheses can be added by external systems or human analysts.
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        # NOTE: The reasoning engine's add_hypothesis method only takes content and confidence
        # The source is hardcoded as "external_hypothesis" in _add_hypothesis_to_context
        operation_id = reasoning_engine.add_hypothesis(
            content=hypothesis.content,
            confidence=hypothesis.confidence
        )
        
        logging.info(f"Hypothesis added: {hypothesis.content[:50]}...")
        
        return HypothesisResponse(
            status="queued",
            operation_id=operation_id,
            message="Hypothesis added successfully"
        )
        
    except Exception as e:
        logging.error(f"Failed to add hypothesis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add hypothesis: {str(e)}")

# NEW: Clear context endpoint
@app.post("/context/clear", response_model=ClearContextResponse, summary="Clear Context")
async def clear_context(request: ClearContextRequest):
    """
    Clear all context and start a new investigation.
    
    Returns an investigation ID that can be used to retrieve results later.
    Context clearing happens asynchronously and may take time if reasoning is in progress.
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        investigation_id = reasoning_engine.clear_context(request.base_name)
        
        logging.info(f"Context clear requested: {investigation_id}")
        
        return ClearContextResponse(
            status="queued",
            investigation_id=investigation_id,
            message=f"Context clear queued - Investigation ID: {investigation_id}"
        )
        
    except Exception as e:
        logging.error(f"Failed to clear context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear context: {str(e)}")

# NEW: Get investigation results endpoint
@app.get("/investigations/{investigation_id}", response_model=InvestigationResults, summary="Get Investigation Results")
async def get_investigation_results(investigation_id: str):
    """
    Retrieve results from a completed investigation.
    
    Returns the final state of hypotheses, evidence, and reasoning cycles
    from when the context was cleared.
    """
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        results = reasoning_engine.get_investigation_results(investigation_id)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"Investigation {investigation_id} not found")
        
        return InvestigationResults(**results)
        
    except Exception as e:
        logging.error(f"Failed to get investigation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get investigation results: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Query Reasoning State")
async def query_context(query: QueryRequest):
    """
    Query the current reasoning state with natural language.
    
    Examples:
    - "What evidence supports enterprise focus?"
    - "Are there any technical issues?"
    - "What contradictions exist in the data?"
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        start_time = datetime.now()
        
        response = reasoning_engine.query_context_sync(
            query=query.query,
            timeout=query.timeout
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if response is None:
            raise HTTPException(status_code=408, detail="Query timeout")
        
        logging.info(f"Query processed: {query.query[:50]}... in {processing_time:.2f}s")
        
        return QueryResponse(
            response=response,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/status", response_model=StatusResponse, summary="Get Engine Status")
async def get_status():
    """
    Get current status and statistics of the reasoning engine.
    
    Includes information about context size, reasoning cycles, and deep thought mode.
    """
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        status = reasoning_engine.get_status_snapshot()
        
        return StatusResponse(
            engine_running=status['engine_running'],
            total_items=status['total_items'],
            hypotheses=status['hypotheses'],
            evidence_items=status['evidence_items'],
            deep_thought_mode=status['deep_thought_mode'],
            deep_thought_duration_seconds=status.get('deep_thought_duration_seconds'),
            reasoning_cycles_total=status['stats']['reasoning_cycles'],
            reasoning_loop_count=status['reasoning_loop_count'],
            max_reasoning_loops=status['max_reasoning_loops'],
            consecutive_sterile_cycles=status['consecutive_sterile_cycles'],
            reasoning_needed=status['reasoning_needed'],
            stats=status['stats']
        )
        
    except Exception as e:
        logging.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/force_stop", summary="Force Stop Reasoning")
async def force_stop_reasoning(request: ForceStopRequest):
    """
    Force the reasoning engine to stop current reasoning cycles.
    
    Useful when the engine appears stuck in deep thought or oscillation.
    The engine will wait for new assertions after stopping.
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        reasoning_engine.force_stop_reasoning(request.reason)
        
        logging.info(f"Reasoning stopped: {request.reason}")
        
        return {
            "status": "success",
            "message": f"Reasoning stopped: {request.reason}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to stop reasoning: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop reasoning: {str(e)}")

# NEW: Configure maximum reasoning loops
@app.post("/config/max_loops", summary="Configure Maximum Reasoning Loops")
async def configure_max_loops(config: ConfigurationRequest):
    """
    Configure the maximum number of reasoning loops per session.
    
    This acts as a failsafe to prevent infinite reasoning loops.
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        reasoning_engine.configure_max_loops(config.max_loops)
        
        logging.info(f"Max reasoning loops configured: {config.max_loops}")
        
        return {
            "status": "success",
            "message": f"Maximum reasoning loops set to {config.max_loops}",
            "max_loops": config.max_loops,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to configure max loops: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure max loops: {str(e)}")

@app.get("/health", summary="Detailed Health Check")
async def health_check():
    """
    Detailed health check including reasoning engine status and performance metrics.
    """
    if not reasoning_engine:
        return {
            "status": "unhealthy",
            "reason": "Reasoning engine not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        status = reasoning_engine.get_status_snapshot()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if not status['engine_running']:
            health_status = "unhealthy"
            issues.append("Engine not running")
        
        if status.get('deep_thought_duration_seconds', 0) > 300:  # 5 minutes
            health_status = "warning"
            issues.append("Extended deep thought")
        
        # Check if reasoning loop counter is near maximum
        if status['reasoning_loop_count'] >= status['max_reasoning_loops'] * 0.8:
            health_status = "warning"
            issues.append(f"High reasoning loop usage: {status['reasoning_loop_count']}/{status['max_reasoning_loops']}")
        
        return {
            "status": health_status,
            "issues": issues,
            "engine_running": status['engine_running'],
            "total_items": status['total_items'],
            "reasoning_cycles": status['stats']['reasoning_cycles'],
            "deep_thought": status['deep_thought_mode'],
            "reasoning_loop_count": status['reasoning_loop_count'],
            "max_reasoning_loops": status['max_reasoning_loops'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Webhook endpoints for common integrations
@app.post("/webhooks/salesforce", summary="Salesforce Webhook")
async def salesforce_webhook(payload: Dict[str, Any]):
    """
    Webhook endpoint for Salesforce integrations.
    Processes Salesforce events and converts them to evidence.
    """
    try:
        # Extract relevant information from Salesforce payload
        event_type = payload.get('type', 'unknown')
        record_data = payload.get('data', {})
        
        # Convert to evidence
        content = f"Salesforce {event_type}: {record_data}"
        
        operation_id = reasoning_engine.add_evidence(
            content=content,
            source="salesforce_webhook",
            confidence=0.9
        )
        
        return {"status": "processed", "operation_id": operation_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.post("/webhooks/monitoring", summary="Monitoring Webhook")
async def monitoring_webhook(payload: Dict[str, Any]):
    """
    Webhook endpoint for monitoring systems (DataDog, New Relic, etc.).
    Processes alerts and metrics as evidence.
    """
    try:
        alert_name = payload.get('alert_name', 'Unknown Alert')
        severity = payload.get('severity', 'info')
        message = payload.get('message', '')
        
        # Determine confidence based on severity
        confidence_map = {
            'critical': 0.95,
            'warning': 0.8,
            'info': 0.6
        }
        confidence = confidence_map.get(severity.lower(), 0.7)
        
        content = f"Monitoring Alert [{severity.upper()}]: {alert_name} - {message}"
        
        operation_id = reasoning_engine.add_evidence(
            content=content,
            source="monitoring_webhook",
            confidence=confidence
        )
        
        return {"status": "processed", "operation_id": operation_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.get("/debug/context", summary="Debug: Get Full Context")
async def debug_get_context():
    """
    Debug endpoint to view the full reasoning context.
    Should be disabled in production for security.
    """
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        # Check if engine has required attributes
        if not hasattr(reasoning_engine, 'context_lock') or not hasattr(reasoning_engine, 'context_items'):
            raise HTTPException(status_code=500, detail="Reasoning engine missing required attributes")
        
        with reasoning_engine.context_lock:
            context_items = []
            for item in reasoning_engine.context_items:
                context_items.append({
                    'content': item.content[:100] + "..." if len(item.content) > 100 else item.content,
                    'type': item.item_type.value,
                    'status': item.status.value,
                    'confidence': item.confidence,
                    'source': item.source,
                    'timestamp': item.timestamp.isoformat(),
                    'reasoning_version': item.reasoning_version,
                    'importance': item.importance,
                    'tags': item.tags
                })
        
        return {
            "context_items": context_items,
            "total_items": len(context_items),
            "context_version": reasoning_engine.context_version if hasattr(reasoning_engine, 'context_version') else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.post("/debug/toggle_full_context", summary="Debug: Toggle Full Context Display")
async def toggle_full_context_display(show: bool = True):
    """
    Toggle display of full context window during reasoning cycles.
    Useful for debugging reasoning behavior.
    """
    if not reasoning_engine or not reasoning_engine.running:
        raise HTTPException(status_code=503, detail="Reasoning engine not available")
    
    try:
        reasoning_engine.toggle_full_context_display(show)
        
        return {
            "status": "success",
            "message": f"Full context display {'ENABLED' if show else 'DISABLED'}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toggle failed: {str(e)}")

# Production startup script
def run_production_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """
    Run the production server with proper configuration.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "webservice:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    import sys
    
    # Simple command-line interface
    if len(sys.argv) > 1:
        if sys.argv[1] == "dev":
            # Development mode with auto-reload
            uvicorn.run("webservice:app", host="127.0.0.1", port=8000, reload=True)
        elif sys.argv[1] == "prod":
            # Production mode
            run_production_server(host="0.0.0.0", port=8000, workers=1)
    else:
        # Default development mode
        print("Starting Reasoning Engine API in development mode")
        print("   API docs: http://localhost:8000/docs")
        print("   Use 'python webservice.py prod' for production mode")
        uvicorn.run("webservice:app", host="127.0.0.1", port=8000, reload=True)
