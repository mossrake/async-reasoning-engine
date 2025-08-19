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
ASYNCHRONOUS REASONING ENGINE - REFERENCE IMPLEMENTATION

This code implements a sophisticated reasoning system that tracks multiple contradictory 
hypotheses over noisy/asynchronous business data, reviving dormant theories when new evidence appears.

ARCHITECTURE OVERVIEW:
- Fast assertion queue: Only updates context, never blocks callers
- Separate reasoning process: Watches context changes, performs LLM reasoning cycles  
- Context monitor: Triggers reasoning when changes detected
- Reasoning cycles: Run asynchronously without blocking new assertions
- Natural completion: Stops when no more meaningful progress can be made

KEY CAPABILITIES:
- Non-blocking evidence/hypothesis ingestion via web APIs
- LLM-based hypothesis generation and confidence updating
- Multiple safety mechanisms prevent infinite reasoning loops
- Detailed progress reporting during deep thought cycles
- Domain-specific reasoning (business analysis, criminal investigation)
- Context compression to manage token limits
- Investigation boundaries with result persistence

SAFETY SYSTEMS:
- Simple loop counter: Hard limit on reasoning cycles (failsafe)
- Oscillation detection: Catches A-B-A-B reasoning patterns
- Sterile cycle detection: Stops when no progress being made  
- Timeout protection: Time-based reasoning limits
- Human override: Manual intervention capability

THREADING MODEL:
- Context thread: Fast processing of assertions and queries
- Reasoning thread: Monitors context changes, performs LLM reasoning
- Single operation queue: Clear boundaries between evidence batches and investigations
"""

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Safety limits - multiple overlapping mechanisms for production robustness
DEFAULT_MAX_REASONING_LOOPS = 10          # Simple counter failsafe (primary)
DEFAULT_MAX_DEEP_THOUGHT_MINUTES = 5      # Time-based safety net
DEFAULT_MAX_STERILE_CYCLES = 3            # Stop when no progress made
OSCILLATION_DETECTION_WINDOW = 6          # Check for A-B-A-B patterns

# Context management  
DEFAULT_MAX_CONTEXT_TOKENS = 4000         # Token budget for LLM calls
COMPRESSION_THRESHOLD_RATIO = 0.8          # When to compress context
ASSERTION_BATCH_SIZE = 5                   # Process in batches during deep thought

# Matching and caching
LLM_MATCHING_CACHE_SIZE = 100             # Cache LLM similarity decisions
HYPOTHESIS_OVERLAP_THRESHOLD = 0.4         # String similarity fallback

# LLM parameters
LLM_TEMPERATURE_MATCHING = 0.1             # Low temp for consistent matching
LLM_TEMPERATURE_REASONING = 0.3            # Higher temp for creative reasoning
LLM_MAX_TOKENS = 2000                      # Response length limit

# ============================================================================
# IMPORTS
# ============================================================================

import os
import threading
import queue
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import dspy
import csv
import requests

# ============================================================================
# CORE DATA STRUCTURES & ENUMS
# ============================================================================

class ItemType(Enum):
    """Types of items that can exist in reasoning context"""
    HYPOTHESIS = "hypothesis"  # Theories about what's happening
    EVIDENCE = "evidence"      # Facts and observations
    PATTERN = "pattern"        # Detected patterns (future use)
    REVIVAL = "revival"        # Hypothesis revival events
    SUMMARY = "summary"        # Compressed summaries

class Status(Enum):
    """Status of hypotheses during reasoning"""
    ACTIVE = "active"          # Currently being considered
    WEAKENED = "weakened"      # Less confident but still viable
    DORMANT = "dormant"        # Set aside but can be revived

class OperationType(Enum):
    """Types of operations that can be queued"""
    ADD_EVIDENCE = "add_evidence"
    ADD_HYPOTHESIS = "add_hypothesis"
    COMPRESS_CONTEXT = "compress_context"
    QUERY_CONTEXT = "query_context"
    CLEAR_CONTEXT = "clear_context"
    SHUTDOWN = "shutdown"

@dataclass
class ContextItem:
    """Individual piece of information in the reasoning context"""
    content: str                              # The actual information
    timestamp: datetime                       # When it was added
    item_type: ItemType                      # What kind of item
    status: Status                           # Current status
    confidence: float                        # How confident we are (0.0-1.0)
    importance: float                        # How important it is (0.0-1.0)
    source: str                              # Where it came from
    tags: List[str]                          # Category tags
    access_count: int = 0                    # How often accessed (for compression)
    reasoning_version: int = 0               # Which reasoning cycle processed this
    
    def to_tuple_string(self) -> str:
        """Convert to tuple format for LLM processing"""
        age_minutes = (datetime.now() - self.timestamp).total_seconds() / 60
        return (f"({self.content}, {self.timestamp.strftime('%H:%M')}, "
                f"conf:{self.confidence:.2f}, {self.status.value}, "
                f"{age_minutes:.0f}min_ago, imp:{self.importance:.2f}, {self.source})")
    
    def token_estimate(self) -> int:
        """Rough estimate of tokens this item will consume"""
        return int(len(self.content.split()) * 1.3)

@dataclass
class ContextOperation:
    """Operation to be processed by the context thread"""
    operation_type: OperationType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    operation_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    result_queue: Optional[queue.Queue] = None

# ============================================================================
# DOMAIN CONFIGURATION
# ============================================================================

class DomainConfig:
    """Domain-specific configuration for different reasoning contexts"""
    
    def __init__(self, domain_type: str = "business"):
        if domain_type == "business":
            self._setup_business_domain()
        elif domain_type == "criminal_investigation":
            self._setup_investigation_domain()
        else:
            self._setup_default_domain()
    
    def _setup_business_domain(self):
        """Configure for business decision support"""
        self.keywords = {
            'sales': ['sales', 'revenue', 'deals'],
            'technical': ['server', 'performance', 'system'],
            'market': ['market', 'competitor'],
            'enterprise': ['enterprise', 'b2b'],
            'smb': ['smb', 'small', 'medium']
        }
        self.context_description = "Business decision support context with focus on strategy, operations, and market analysis"
        self.hypothesis_keywords = ['enterprise', 'smb', 'customer', 'revenue', 'performance', 'market', 'sales']
        self.executive_role = "CEO/senior leadership team"
        self.output_format = "executive briefing"
        
        self.hypothesis_instructions = """
Generate 3-5 detailed BUSINESS hypotheses from evidence. Focus on:
- Market conditions and competitive dynamics
- Operational efficiency and performance issues  
- Strategic business decisions and outcomes
- Revenue, costs, and profitability factors

Each hypothesis should:
- Provide a comprehensive explanation of business patterns
- Include specific details about mechanisms and relationships
- Connect multiple pieces of evidence into a coherent narrative
- Explain strategic implications with supporting rationale
- Be substantive enough to guide meaningful decision-making

Elaborate as needed to fully develop each hypothesis - detailed analysis is preferred over brevity.
"""
    
    def _setup_investigation_domain(self):
        """Configure for criminal investigation"""
        self.keywords = {
            'evidence': ['forensic', 'dna', 'fingerprint', 'witness', 'crime_scene', 'blood', 'weapon'],
            'temporal': ['timeline', 'alibi', 'when', 'time', 'midnight', 'evening', 'morning'],
            'relationship': ['suspect', 'victim', 'family', 'friend', 'spouse', 'partner', 'business'],
            'location': ['scene', 'location', 'address', 'where', 'apartment', 'building', 'room'],
            'motive': ['motive', 'reason', 'why', 'conflict', 'inheritance', 'money', 'revenge', 'jealousy'],
            'method': ['entry', 'locked', 'forced', 'window', 'door', 'weapon', 'strangled', 'shot']
        }
        self.context_description = "Criminal investigation context focusing on evidence analysis, suspect identification, and case resolution"
        self.hypothesis_keywords = ['suspect', 'victim', 'evidence', 'motive', 'witness', 'crime', 'forensic']
        self.executive_role = "detective/investigation team"      
        self.output_format = "case summary"
        
        self.hypothesis_instructions = """
You are analyzing a CRIMINAL CASE. Generate 3-5 SPECIFIC detailed hypotheses naming WHO did it and WHY.

IT IS VERY IMPORTANT to Use ONLY the real names and people mentioned in the evidence. Do NOT invent fictional characters.
Build a roster of potential suspects.

EVIDENCE ANALYSIS:
- Read the evidence carefully for actual names, relationships, and motives
- If evidence says "business partner John Smith" - use "John Smith", not "the business partner"
- If evidence mentions specific usernames or identities - use those exact identities
- If evidence lists family members by name - use their real names
- Connect specific people to specific motives from the evidence

DETECTIVE PRINCIPLES:
- Always suspect the closest person first (family, partners, business associates)
- Follow the money (who benefits financially?)
- Use DNA, forensics, and timeline to eliminate/confirm suspects
- Name the specific person and their specific motive

Each hypothesis should:
- Provide detailed explanation of who committed the crime and why
- Include comprehensive analysis of evidence supporting this theory
- Explain the timeline and method used
- Address potential counterarguments or alternative explanations
- Connect multiple pieces of evidence into a coherent case narrative

Generate hypotheses that connect real people from evidence to real motives:
- "[Real name from evidence] killed the victim because [specific detailed motive from evidence]"
- "The DNA evidence points to [specific person mentioned] who had [specific opportunity and detailed reasoning]"

Focus on WHO did it using actual evidence, with detailed supporting analysis.
"""
    
    def _setup_default_domain(self):
        """Fallback domain configuration"""
        self.keywords = {
            'analysis': ['analysis', 'assessment', 'evaluation'],
            'evidence': ['evidence', 'data', 'information'],
            'conclusion': ['conclusion', 'result', 'finding'],
            'pattern': ['pattern', 'trend', 'correlation']
        }
        self.context_description = "General analysis context for evidence evaluation and hypothesis generation"
        self.hypothesis_keywords = ['analysis', 'evidence', 'conclusion', 'assessment', 'evaluation', 'pattern']
        self.executive_role = "analyst/decision maker"
        self.output_format = "analytical report"
        
        self.hypothesis_instructions = """
Generate 3-5 detailed hypotheses from the available evidence. Focus on:
- Patterns and correlations in the data
- Cause and effect relationships
- Potential explanations for observed phenomena
- Alternative interpretations of the evidence

Each hypothesis should:
- Provide comprehensive explanation of what you believe is happening
- Include detailed reasoning connecting evidence to conclusions
- Explain underlying mechanisms that would produce observed patterns
- Be substantive enough to guide further analysis and decision-making

Elaborate as needed to fully develop each hypothesis - thorough analysis is preferred over brevity.
"""

# ============================================================================
# LLM INTEGRATION & AZURE OPENAI
# ============================================================================

class LLMMatchingCache:
    """Cache for LLM-based hypothesis similarity decisions"""
    
    def __init__(self, max_size: int = LLM_MATCHING_CACHE_SIZE):
        self.cache = {}
        self.max_size = max_size
    
    def get_cache_key(self, text_a: str, text_b: str) -> str:
        """Create deterministic cache key"""
        combined = f"{text_a.lower()}|||{text_b.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def get(self, text_a: str, text_b: str) -> Optional[bool]:
        """Get cached similarity result"""
        key = self.get_cache_key(text_a, text_b)
        # Check both orders since matching should be symmetric
        reverse_key = self.get_cache_key(text_b, text_a)
        
        if key in self.cache:
            return self.cache[key]
        elif reverse_key in self.cache:
            return self.cache[reverse_key]
        
        return None
    
    def set(self, text_a: str, text_b: str, result: bool):
        """Cache similarity result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_cache_key(text_a, text_b)
        self.cache[key] = result

def setup_dspy():
    """Setup DSPy with Azure OpenAI"""
    required_vars = ['AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    try:
        lm = dspy.LM(
            model=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
        )
        
        dspy.configure(lm=lm)
        print("DSPy configured with Azure OpenAI")
        return True
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup DSPy: {e}")

class LLMContextManager(dspy.Module):
    """LLM integration manager - combines DSPy and direct API calls"""
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__()
        self.domain_config = domain_config
        self.matching_cache = LLMMatchingCache()
        
        # Azure OpenAI compatible DSPy signatures (no instructions parameter)
        self.reason_about_context = dspy.ChainOfThought(
            "context_summary, new_items -> reasoning_analysis"
        )
        
        self.manage_context = dspy.ChainOfThought(
            "current_context_items, compression_needed -> context_decisions"
        )
        
        self.analyze_query = dspy.ChainOfThought(
            "context_summary, user_query -> query_response"
        )
        
        self.generate_hypotheses = dspy.ChainOfThought(
            "evidence_items, business_context -> hypothesis_suggestions"
        )

    def check_hypothesis_equivalence(self, hypothesis_a: str, hypothesis_b: str) -> bool:
        """Use LLM to determine if two hypotheses are substantially the same"""
        
        # Check cache first
        cached_result = self.matching_cache.get(hypothesis_a, hypothesis_b)
        if cached_result is not None:
            return cached_result
        
        # Direct Azure OpenAI call for guaranteed JSON response
        prompt = f"""You are comparing two {self.domain_config.context_description.lower()} hypotheses to determine if they represent substantially the same theory or conclusion.

HYPOTHESIS A: {hypothesis_a}

HYPOTHESIS B: {hypothesis_b}

INSTRUCTIONS:
Determine if these hypotheses are substantially equivalent by considering:

1. CORE THESIS: Do they make the same fundamental claim about what is happening?
2. CAUSAL RELATIONSHIPS: Do they identify the same cause-and-effect patterns?
3. IMPLICATIONS: Do they suggest similar conclusions or outcomes?
4. SCOPE: Are they addressing the same area/problem?

IGNORE differences in:
- Level of detail (one might be more elaborate)
- Specific wording or phrasing
- Order of information presented
- Minor variations in supporting evidence cited

EXAMPLES OF EQUIVALENCE:
- "Sales exceeded targets" vs "Q4 sales performance surpassed quarterly goals by 30%"
- "CRM improved efficiency" vs "New customer relationship management system streamlined sales pipeline"
- "Market competition increased" vs "Competitive pressure intensified due to new entrants"

EXAMPLES OF NON-EQUIVALENCE:
- "Sales exceeded targets" vs "Marketing campaign failed"
- "CRM improved efficiency" vs "Customer satisfaction declined"
- "Enterprise segment grew" vs "SMB segment grew"

Respond with JSON in this exact format:
{{
  "equivalent": true/false,
  "confidence": 0.X,
  "reasoning": "brief explanation of why they are/aren't equivalent"
}}"""

        try:
            result = self._call_azure_openai_json(prompt, LLM_TEMPERATURE_MATCHING)
            decision = json.loads(result)
            
            # Return True if equivalent with reasonable confidence
            is_equivalent = decision.get("equivalent", False) and decision.get("confidence", 0) > 0.7
            
            # Cache the result
            self.matching_cache.set(hypothesis_a, hypothesis_b, is_equivalent)
            
            return is_equivalent
            
        except Exception as e:
            print(f"   LLM matching failed: {e}, falling back to string matching")
            # Fallback to existing string matching if LLM fails
            result = self._fallback_string_match(hypothesis_a, hypothesis_b)
            self.matching_cache.set(hypothesis_a, hypothesis_b, result)
            return result

    def _fallback_string_match(self, existing_content: str, llm_content: str) -> bool:
        """Fallback string matching if LLM matching fails"""
        existing_lower = existing_content.lower()
        llm_lower = llm_content.lower()
        
        # Simple substring check
        if llm_lower in existing_lower or existing_lower in llm_lower:
            return True
        
        # Basic word overlap
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'that', 'this'}
        existing_words = set(existing_lower.split()) - stop_words
        llm_words = set(llm_lower.split()) - stop_words
        
        if len(existing_words) == 0 or len(llm_words) == 0:
            return False
        
        intersection = existing_words.intersection(llm_words)
        similarity = len(intersection) / max(len(existing_words), len(llm_words))
        
        return similarity > HYPOTHESIS_OVERLAP_THRESHOLD

    def generate_initial_hypotheses(self, evidence_items: str) -> str:
        """LLM generates initial hypotheses from evidence using domain config"""
        
        hypothesis_prompt = f"""
{self.domain_config.hypothesis_instructions}

Format each as: "HYPOTHESIS: [detailed explanation of what you believe is happening, including mechanisms, relationships, and implications] | CONFIDENCE: [0.X] | REASONING: [comprehensive justification connecting evidence to conclusion]"

Provide thorough, substantive content in each field. The hypothesis field should contain a complete explanation, not just a brief statement.

EVIDENCE ITEMS:
{evidence_items}

CONTEXT: {self.domain_config.context_description}
"""
        
        hypotheses = self.generate_hypotheses(
            evidence_items=hypothesis_prompt,
            business_context=self.domain_config.context_description
        )
        return hypotheses.hypothesis_suggestions

    def reason_about_context_changes(self, context_summary: str, new_items: str) -> str:
        """Direct Azure OpenAI call for reasoning about context changes"""
        
        # Explicit instruction to update confidence based on evidence
        prompt = f"""You are analyzing {self.domain_config.context_description.lower()} hypotheses based on new evidence. Your job is to UPDATE confidence scores based on how the new evidence supports or contradicts each hypothesis.
    
CURRENT HYPOTHESES WITH CONFIDENCE SCORES:
{context_summary}

NEW EVIDENCE TO ANALYZE:
{new_items}

INSTRUCTIONS:
1. For each hypothesis, determine if the new evidence SUPPORTS, CONTRADICTS, or is NEUTRAL
2. INCREASE confidence (by 0.1-0.3) for hypotheses supported by evidence
3. DECREASE confidence (by 0.1-0.3) for hypotheses contradicted by evidence  
4. Keep confidence unchanged only if evidence is truly neutral
5. Provide detailed reasoning for confidence changes

RESPOND WITH JSON in this EXACT format:
{{
  "hypotheses": [
    {{
      "statement": "exact hypothesis text",
      "confidence": updated_confidence_score,
      "status": "active",
      "reasoning": "detailed explanation of why confidence changed based on evidence"
    }}
  ],
  "new_hypotheses": []
}}

IMPORTANT: Confidence scores MUST be different from input if evidence is relevant. Do not just echo the same confidence values."""
    
        return self._call_azure_openai_json(prompt, LLM_TEMPERATURE_REASONING)

    def _call_azure_openai_json(self, prompt: str, temperature: float) -> str:
        """Make direct Azure OpenAI API call with JSON mode"""
        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_VERSION')}"
        
        headers = {
            "api-key": os.getenv("AZURE_OPENAI_KEY"),
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": LLM_MAX_TOKENS
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

# ============================================================================
# MAIN REASONING ENGINE
# ============================================================================

class AsyncReasoningEngine:
    """
    ASYNCHRONOUS REASONING ENGINE
    
    ARCHITECTURE:
    - Fast assertion queue that only updates context (never blocks callers)
    - Separate reasoning process that watches context changes
    - Reasoning involves rewriting the context, and adding to it
    - Reasoning cycles run asynchronously without blocking assertions
    - Context monitor triggers reasoning when changes detected
    - Detailed progress reporting during deep thought cycles
    
    THREADING MODEL:
    - Context thread: Handles assertions, queries, and context operations
    - Reasoning thread: Monitors context changes and performs LLM reasoning
    - Single operation queue with clear boundaries
    
    SAFETY SYSTEMS:
    - Simple loop counter failsafe to prevent infinite reasoning / ruminating / oscillation
    - Oscillation detection with hash history
    - Sterile cycle detection (no meaningful progress)
    - Timeout protection and human override capability
    """
    
    def __init__(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS, domain: str = "business"):
        # Domain configuration
        self.domain_config = DomainConfig(domain)
        
        # Setup LLM integration
        setup_dspy()
        self.llm_manager = LLMContextManager(self.domain_config)  

        # Core state
        self.context_items: List[ContextItem] = []
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = int(max_context_tokens * COMPRESSION_THRESHOLD_RATIO)
        self.running = False
        
        # Threading components
        self.queued_assertions = queue.Queue()  # Single queue for all operations
        self.context_thread = None              # Fast context updates
        self.reasoning_thread = None            # Slow reasoning cycles
        
        # Context change tracking
        self.context_version = 0          # Incremented on each context change
        self.last_reasoned_version = 0    # Last version that was reasoned about
        self.context_change_event = threading.Event()  # Signals reasoning needed
        
        # Safety mechanisms (multiple overlapping systems for production robustness)
        self.reasoning_loop_count = 0     # Simple counter for current reasoning session
        self.max_reasoning_loops = DEFAULT_MAX_REASONING_LOOPS     # Hard stop after N loops
        self.consecutive_sterile_cycles = 0
        self.max_sterile_cycles = DEFAULT_MAX_STERILE_CYCLES       # Stop after N cycles with no context changes
        self.last_context_hash = None       # Hash of context after last reasoning cycle
        self.context_hash_history = []      # Track hash history for oscillation detection
        self.oscillation_detection_window = OSCILLATION_DETECTION_WINDOW
        
        # Deep thought monitoring
        self.deep_thought_mode = False
        self.deep_thought_start = None
        self.max_deep_thought_minutes = DEFAULT_MAX_DEEP_THOUGHT_MINUTES
        self.human_override_event = threading.Event()  # Allow human interruption
        self.show_full_context = False      # Option to show full context each cycle
        
        # Investigation tracking
        self.investigation_sequence = 0          # Sequence counter for investigation IDs
        
        # Thread synchronization
        self.context_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'assertions_processed': 0,
            'reasoning_cycles': 0,
            'hypotheses_generated': 0,
            'revivals_detected': 0,
            'compressions_performed': 0,
            'context_changes': 0
        }

    # ========================================================================
    # PUBLIC API - Non-blocking operations for web API integration
    # ========================================================================
    
    def start(self):
        """Start both context processing and reasoning monitoring threads"""
        if self.running:
            return
        
        self.running = True
        
        # Start fast context processing thread
        self.context_thread = threading.Thread(target=self._context_processing_loop, daemon=True)
        self.context_thread.start()
        
        # Start separate reasoning monitoring thread  
        self.reasoning_thread = threading.Thread(target=self._reasoning_monitoring_loop, daemon=True)
        self.reasoning_thread.start()
        
        print("Async reasoning engine started")
        print("   Context processor: handles assertions without blocking")
        print("   Reasoning monitor: watches for context changes")
    
    def stop(self):
        """Stop all processing gracefully"""
        if not self.running:
            return
        
        print("Stopping async reasoning engine...")
        
        # Signal shutdown
        shutdown_op = ContextOperation(operation_type=OperationType.SHUTDOWN, data={})
        self.queued_assertions.put(shutdown_op)
        self.context_change_event.set()  # Wake up reasoning thread
        
        # Wait for threads
        if self.context_thread and self.context_thread.is_alive():
            self.context_thread.join(timeout=5.0)
        if self.reasoning_thread and self.reasoning_thread.is_alive():
            self.reasoning_thread.join(timeout=5.0)
        
        self.running = False
        print("Async reasoning engine stopped")
    
    def add_evidence(self, content: str, source: str, confidence: float = 0.8) -> str:
        """Add evidence to the reasoning context (non-blocking)"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        operation = ContextOperation(
            operation_type=OperationType.ADD_EVIDENCE,
            data={'content': content, 'source': source, 'confidence': confidence}
        )
        
        self.queued_assertions.put(operation)
        print(f"Evidence queued: {content[:50]}...")
        return operation.operation_id
    
    def add_hypothesis(self, content: str, confidence: float = 0.6, source: str = "external_hypothesis") -> str:
        """Add hypothesis to the reasoning context (non-blocking)"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        operation = ContextOperation(
            operation_type=OperationType.ADD_HYPOTHESIS,
            data={'content': content, 'confidence': confidence, 'source': source}
        )
        
        self.queued_assertions.put(operation)
        print(f"Hypothesis queued: {content[:50]}... (from {source})")
        return operation.operation_id
  
    def clear_context(self, base_name: str = "investigation") -> str:
        """Clear all context and start new investigation (non-blocking)"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        # Generate unique investigation ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.investigation_sequence += 1
        investigation_id = f"{base_name}_{timestamp}_{self.investigation_sequence:03d}"
        
        operation = ContextOperation(
            operation_type=OperationType.CLEAR_CONTEXT,
            data={
                'reason': f"Investigation: {base_name}",
                'investigation_id': investigation_id
            }
        )
        
        self.queued_assertions.put(operation)
        print(f"Context clear queued: {investigation_id}")
        return investigation_id
       
    def query_context_sync(self, query: str, timeout: float = 30.0) -> Optional[str]:
        """Query reasoning state synchronously (blocks until response)"""
        if not self.running:
            return "Engine not running"
        
        result_queue = queue.Queue()
        operation = ContextOperation(
            operation_type=OperationType.QUERY_CONTEXT,
            data={'query': query},
            result_queue=result_queue
        )
        
        self.queued_assertions.put(operation)
        
        try:
            result = result_queue.get(timeout=timeout)
            return result.get('response', 'Query failed')
        except queue.Empty:
            return f"Query timeout after {timeout}s"

    # ========================================================================
    # MONITORING & CONTROL API
    # ========================================================================

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Get current engine status for monitoring"""
        with self.context_lock:
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
            evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE]
            
            # Deep thought status
            deep_thought_duration = None
            if self.deep_thought_mode and self.deep_thought_start:
                deep_thought_duration = (datetime.now() - self.deep_thought_start).total_seconds()
            
            return {
                'engine_running': self.running,
                'total_items': len(self.context_items),
                'hypotheses': len(hypotheses),
                'evidence_items': len(evidence),
                'context_version': self.context_version,
                'last_reasoned_version': self.last_reasoned_version,
                'reasoning_needed': self.context_version > self.last_reasoned_version,
                'estimated_tokens': sum(item.token_estimate() for item in self.context_items),
                'deep_thought_mode': self.deep_thought_mode,
                'deep_thought_duration_seconds': deep_thought_duration,
                'consecutive_sterile_cycles': self.consecutive_sterile_cycles,
                'reasoning_loop_count': self.reasoning_loop_count,
                'max_reasoning_loops': self.max_reasoning_loops,
                'stats': self.stats.copy()
            }

    def get_context_summary(self) -> Dict[str, Any]:
        """Get detailed context summary for debugging"""
        with self.context_lock:
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
            evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE]
            
            context_items = []
            for item in self.context_items:
                # Truncate content for summary
                content = item.content[:100] + "..." if len(item.content) > 100 else item.content
                
                context_items.append({
                    'content': content,
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
                'context_items': context_items,
                'total_items': len(self.context_items),
                'hypotheses_count': len(hypotheses),
                'evidence_count': len(evidence),
                'context_version': self.context_version,
                'estimated_tokens': sum(item.token_estimate() for item in self.context_items)
            }

    def force_stop_reasoning(self, reason: str = "Human override"):
        """Force reasoning to stop immediately"""
        print(f"FORCED REASONING STOP: {reason}")
        print(f"   Was on loop {self.reasoning_loop_count} of {self.max_reasoning_loops}")
        self.human_override_event.set()
        self.deep_thought_mode = False

    def configure_max_loops(self, max_loops: int):
        """Configure maximum reasoning loops per session"""
        self.max_reasoning_loops = max_loops
        print(f"Maximum reasoning loops set to: {max_loops}")

    def toggle_full_context_display(self, show: bool = True):
        """Toggle display of full context window at each reasoning cycle"""
        self.show_full_context = show
        print(f"Full context display: {'ENABLED' if show else 'DISABLED'}")

    def reconfigure_domain(self, new_domain: str) -> bool:
        """Reconfigure domain without full restart"""
        try:
            # Validate domain
            if new_domain not in ["business", "criminal_investigation"]:
                raise ValueError(f"Invalid domain: {new_domain}")
            
            # Update domain config
            old_domain = getattr(self.domain_config, 'context_description', 'unknown')
            self.domain_config = DomainConfig(new_domain)
            
            # Update LLM manager with new domain config
            self.llm_manager.domain_config = self.domain_config
            
            print(f"Domain reconfigured from '{old_domain}' to '{new_domain}'")
            return True
            
        except Exception as e:
            print(f"Domain reconfiguration failed: {e}")
            return False

    def get_investigation_results(self, investigation_id: str) -> Dict[str, Any]:
        """Get investigation results by ID (placeholder for SQLite integration)"""
        # TODO: Retrieve from SQLite database
        print(f"STUB: Retrieving results for investigation {investigation_id}")
        return {
            "investigation_id": investigation_id,
            "status": "completed",
            "hypotheses": [],
            "evidence_count": 0,
            "confidence_summary": {},
            "stub": True
        }

    # ========================================================================
    # CONTEXT PROCESSING THREAD - Fast, non-blocking operations
    # ========================================================================
    
    def _context_processing_loop(self):
        """Main context processing loop - handles operations from queue"""
        print("Context processing started (single queue)")
        
        while self.running:
            try:
                # Process operations with clear boundary logic
                operations_to_process = self._collect_operations_before_boundary()
                
                if operations_to_process:
                    for operation in operations_to_process:
                        if operation.operation_type == OperationType.SHUTDOWN:
                            return
                        
                        print(f"Processing: {operation.operation_type.value}")
                        self._process_context_operation(operation)
                        
                        # Signal context change for evidence/hypothesis operations
                        if operation.operation_type in [OperationType.ADD_EVIDENCE, OperationType.ADD_HYPOTHESIS]:
                            self._signal_context_change(is_external=True)
                else:
                    # No operations to process, wait briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Context processing error: {e}")
                time.sleep(1)
        
        print("Context processing stopped")

    def _collect_operations_before_boundary(self) -> List[ContextOperation]:
        """Collect operations to process before next clear boundary"""
        
        # Check if clear is at head of queue (after deep thought completes)
        if not self.deep_thought_mode and not self.queued_assertions.empty():
            # Peek at next operation without removing it
            try:
                next_op = self.queued_assertions.get_nowait()
                
                if next_op.operation_type == OperationType.CLEAR_CONTEXT:
                    # Process clear immediately
                    print("Processing clear at head of queue")
                    self._process_context_operation(next_op)
                    return []
                else:
                    # Put it back, we'll process in batch
                    self.queued_assertions.put(next_op)
            except queue.Empty:
                pass
        
        # Collect operations before first clear boundary
        operations_to_process = []
        clear_found = False
        temp_operations = []
        
        # Scan queue to find operations before first clear
        while not self.queued_assertions.empty() and not clear_found:
            try:
                operation = self.queued_assertions.get_nowait()
                temp_operations.append(operation)
                
                if operation.operation_type == OperationType.SHUTDOWN:
                    # Process shutdown immediately
                    operations_to_process.append(operation)
                    break
                elif operation.operation_type == OperationType.CLEAR_CONTEXT:
                    # Found clear - stop processing, put clear back
                    clear_found = True
                    self.queued_assertions.put(operation)
                else:
                    # Regular operation - process it
                    operations_to_process.append(operation)
                    
            except queue.Empty:
                break
        
        # Put back any operations we couldn't process yet
        for op in temp_operations:
            if op not in operations_to_process and op.operation_type != OperationType.CLEAR_CONTEXT:
                self.queued_assertions.put(op)
        
        return operations_to_process
    
    def _process_context_operation(self, operation: ContextOperation):
        """Process individual context operation"""
        
        try:
            if operation.operation_type == OperationType.ADD_EVIDENCE:
                self._add_evidence_to_context(operation.data)
            
            elif operation.operation_type == OperationType.ADD_HYPOTHESIS:
                self._add_hypothesis_to_context(operation.data)
            
            elif operation.operation_type == OperationType.CLEAR_CONTEXT:
                self._clear_context(operation.data)
            
            elif operation.operation_type == OperationType.QUERY_CONTEXT:
                self._process_query(operation)
            
            # Send success if needed
            if operation.result_queue:
                operation.result_queue.put({'success': True})
                
        except Exception as e:
            print(f"Context operation failed: {e}")
            if operation.result_queue:
                operation.result_queue.put({'success': False, 'error': str(e)})

    def _add_evidence_to_context(self, data: Dict[str, Any]):
        """Add evidence item to context"""
        evidence_item = ContextItem(
            content=data['content'],
            timestamp=datetime.now(),
            item_type=ItemType.EVIDENCE,
            status=Status.ACTIVE,
            confidence=data['confidence'],
            importance=self._assess_importance(data['content']),
            source=data['source'],
            tags=self._extract_tags(data['content']),
            reasoning_version=0  # Not yet reasoned about
        )
        
        with self.context_lock:
            self.context_items.append(evidence_item)
            self.stats['assertions_processed'] += 1
        
        print(f"Evidence added: {data['content'][:50]}... (from {data['source']})")
    
    def _add_hypothesis_to_context(self, data: Dict[str, Any]):
        """Add hypothesis item to context"""
        hypothesis_item = ContextItem(
            content=data['content'],
            timestamp=datetime.now(),
            item_type=ItemType.HYPOTHESIS,
            status=Status.ACTIVE,
            confidence=data['confidence'],
            importance=0.9,
            source=data.get('source', 'external_hypothesis'),
            tags=["hypothesis"],
            reasoning_version=0
        )
        
        with self.context_lock:
            self.context_items.append(hypothesis_item)
        
        print(f"Hypothesis added: {data['content'][:50]}... (from {data.get('source', 'unknown')})")
    
    def _clear_context(self, data: Dict[str, Any]):
        """Clear all context items and store investigation results"""
        reason = data.get('reason', 'Unknown')
        investigation_id = data.get('investigation_id')
        
        # Capture current investigation results before clearing
        if investigation_id:
            current_results = self._capture_investigation_results()
            self._store_investigation_results(investigation_id, current_results)
        
        with self.context_lock:
            old_count = len(self.context_items)
            self.context_items.clear()
            self.context_version += 1
            # Reset reasoning state after clear
            self.last_reasoned_version = self.context_version
            self.reasoning_loop_count = 0
            self.consecutive_sterile_cycles = 0
            self.context_hash_history.clear()
            
        print(f"Context cleared: {old_count} items removed ({reason})")
        if investigation_id:
            print(f"Investigation results stored: {investigation_id}")

    def _process_query(self, operation: ContextOperation):
        """Process query using current context with executive-focused responses"""
        query = operation.data['query']
        
        with self.context_lock:
            # Get current state for executive summary
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
            evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE]
            
            # Find top hypothesis
            top_hypothesis = max(hypotheses, key=lambda x: x.confidence) if hypotheses else None
            
            # Categorize evidence by source/type
            evidence_by_source = {}
            for e in evidence:
                if e.source not in evidence_by_source:
                    evidence_by_source[e.source] = []
                evidence_by_source[e.source].append(e)
        
        try:
            # Build domain-focused prompt
            executive_prompt = f"""You are providing a {self.domain_config.output_format} to a {self.domain_config.executive_role}. 
    
CURRENT SITUATION ANALYSIS:
- Top conclusion: {top_hypothesis.content if top_hypothesis else 'No clear conclusions yet'}
- Confidence level: {top_hypothesis.confidence:.0%} if top_hypothesis else 'N/A'
- Supporting evidence from {len(evidence_by_source)} data sources
- {len(hypotheses)} competing theories analyzed

KEY EVIDENCE:
{chr(10).join([f"\u2022 {e.content}" for e in evidence[:5]])}

QUERY: {query}

INSTRUCTIONS:
Provide a crisp, executive-ready response with:
1. BOTTOM LINE UP FRONT: One clear conclusion/recommendation
2. CONFIDENCE LEVEL: Specific percentage or "high/medium/low" with reasoning
3. KEY SUPPORTING FACTS: 2-3 bullet points of strongest evidence
4. IMPLICATIONS: What this means for strategy/operations
5. NEXT STEPS: What should be done based on this analysis

Keep it executive-appropriate: confident, specific, actionable. Avoid hedging language like "given the evidence" or "appears to suggest." Make clear statements backed by data."""
    
            context_summary = self._build_context_summary()
            
            response = self.llm_manager.analyze_query(
                context_summary=context_summary,
                user_query=executive_prompt
            )
            
            if operation.result_queue:
                operation.result_queue.put({'response': response.query_response})
                
        except Exception as e:
            if operation.result_queue:
                operation.result_queue.put({'response': f"Analysis failed: {e}"})

    def _signal_context_change(self, is_external=True):
        """Signal that context has changed to trigger reasoning"""
        with self.context_lock:
            self.context_version += 1
            # Reset sterile cycles when new external assertions arrive
            if is_external:
                self.consecutive_sterile_cycles = 0
            self.stats['context_changes'] += 1
        
        # Wake up reasoning thread
        self.context_change_event.set()

    # ========================================================================
    # REASONING MONITORING THREAD - Watches for context changes
    # ========================================================================
    
    def _reasoning_monitoring_loop(self):
        """Reasoning monitor loop - watches for context changes and triggers reasoning"""
        print("Reasoning monitor started (watches for context changes)")
        
        while self.running:
            try:
                # Wait for context changes or timeout
                self.context_change_event.wait(timeout=5.0)
                
                if not self.running:
                    break
                
                # Check if reasoning is needed
                if self.context_version > self.last_reasoned_version:
                    self._enter_deep_thought_if_needed()
                    
                    reasoning_result = self._perform_reasoning_cycle()
                    
                    # Check stop conditions
                    if self._should_stop_reasoning(reasoning_result):
                        self._exit_deep_thought()
                else:
                    # No reasoning needed, exit deep thought
                    if self.deep_thought_mode:
                        self._exit_deep_thought()
                
                # Clear the event
                self.context_change_event.clear()
                
            except Exception as e:
                print(f"Reasoning monitor error: {e}")
                time.sleep(1)
        
        print("Reasoning monitor stopped")

    def _enter_deep_thought_if_needed(self):
        """Enter deep thought mode if multiple cycles expected"""
        if not self.deep_thought_mode:
            self.deep_thought_mode = True
            self.deep_thought_start = datetime.now()
            self.reasoning_loop_count = 0  # Reset counter for new session
            print("ENTERING DEEP THOUGHT MODE...")

    def _exit_deep_thought(self):
        """Exit deep thought mode and generate summary"""
        print("EXITING DEEP THOUGHT MODE - reasoning complete")
        self._generate_deep_thought_summary()
        self.deep_thought_mode = False
        print("Deep thought complete - context processing will resume")

    def _should_stop_reasoning(self, reasoning_result: Dict[str, Any]) -> bool:
        """Check if reasoning should stop based on various conditions"""
        
        # Check reasoning result
        if reasoning_result['should_stop_reasoning']:
            print(f"Reasoning stopped: {reasoning_result['stop_reason']}")
            return True
        
        # Check for human override
        if self.human_override_event.is_set():
            print("Reasoning stopped: Human override requested")
            self.human_override_event.clear()
            return True
        
        # Check deep thought timeout
        if self.deep_thought_mode and self.deep_thought_start:
            deep_thought_duration = (datetime.now() - self.deep_thought_start).total_seconds()
            if deep_thought_duration > (self.max_deep_thought_minutes * 60):
                print(f"DEEP THOUGHT TIMEOUT after {deep_thought_duration/60:.1f} minutes")
                print("Pausing reasoning - waiting for external intervention or new assertions")
                return True
        
        return False

    # ========================================================================
    # REASONING CYCLE LOGIC - Core LLM-based reasoning
    # ========================================================================
    
    def _perform_reasoning_cycle(self) -> Dict[str, Any]:
        """Perform single reasoning cycle with comprehensive safety checks"""
        
        # Increment and check simple loop counter (primary failsafe)
        self.reasoning_loop_count += 1
        
        print(f"\nREASONING CYCLE {self.reasoning_loop_count} STARTED")
        print(f"   Loop {self.reasoning_loop_count} of {self.max_reasoning_loops} max")
        
        # Display context if enabled
        if self.show_full_context:
            self._display_full_context_window()

        # Check failsafe first
        if self.reasoning_loop_count >= self.max_reasoning_loops:
            return {
                'should_stop_reasoning': True,
                'stop_reason': f'FAILSAFE: Maximum reasoning loops reached ({self.max_reasoning_loops})',
                'context_changed': False,
                'failsafe_triggered': True
            }
        
        # Pre-reasoning setup and validation
        before_state, context_hash_before = self._setup_reasoning_cycle()
        
        if not self._validate_reasoning_preconditions():
            return {
                'should_stop_reasoning': True,
                'stop_reason': 'Reasoning preconditions not met',
                'context_changed': False
            }
        
        # Execute reasoning logic
        try:
            self._execute_reasoning_operations()
            
            # Post-reasoning analysis
            changes = self._track_reasoning_changes(before_state)
            self._report_reasoning_progress(self.reasoning_loop_count, changes)
            
            # Check for completion conditions
            return self._evaluate_reasoning_completion(context_hash_before)
            
        except Exception as e:
            print(f"Reasoning cycle failed: {e}")
            return {
                'should_stop_reasoning': True,
                'stop_reason': f'Reasoning error: {e}',
                'context_changed': False
            }

    def _setup_reasoning_cycle(self) -> Tuple[List[ContextItem], str]:
        """Setup reasoning cycle and capture before state"""
        
        # Capture before state for change tracking
        with self.context_lock:
            before_items = [
                ContextItem(
                    content=item.content,
                    timestamp=item.timestamp,
                    item_type=item.item_type,
                    status=item.status,
                    confidence=item.confidence,
                    importance=item.importance,
                    source=item.source,
                    tags=item.tags.copy(),
                    access_count=item.access_count,
                    reasoning_version=item.reasoning_version
                ) for item in self.context_items if item.item_type == ItemType.HYPOTHESIS
            ]
        
        # Compute context hash before reasoning
        context_hash_before = self._compute_context_hash()
        
        return before_items, context_hash_before

    def _validate_reasoning_preconditions(self) -> bool:
        """Check if reasoning should proceed"""
        with self.context_lock:
            cycle_start_version = self.context_version
            new_items = [item for item in self.context_items 
                          if item.reasoning_version < cycle_start_version]
            
            if not new_items:
                return False
            
            # Don't reason if context is empty
            if len(self.context_items) == 0:
                return False
            
            # Log what we're processing
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
            print(f"   Processing {len(new_items)} new items")
            print(f"   Current hypotheses: {len(hypotheses)}")
            print(f"   Consecutive sterile cycles: {self.consecutive_sterile_cycles}")
            
            return True

    def _execute_reasoning_operations(self):
        """Execute the core reasoning operations"""
        with self.context_lock:
            cycle_start_version = self.context_version
            new_items = [item for item in self.context_items 
                          if item.reasoning_version < cycle_start_version]
            context_summary = self._build_context_summary()
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
        
        # Generate initial hypotheses if none exist
        if not hypotheses and new_items:
            print(f"   Generating initial hypotheses from evidence...")
            self._generate_initial_hypotheses(new_items)
        
        # Reason about new evidence against existing hypotheses
        elif hypotheses and new_items:
            print(f"   Analyzing new evidence against existing hypotheses...")
            self._reason_about_new_items(context_summary, new_items)
        
        # Perform compression if needed
        total_tokens = sum(item.token_estimate() for item in self.context_items)
        if total_tokens > self.compression_threshold:
            print(f"   Performing context compression...")
            self._perform_compression()
        
        # Mark items as processed
        with self.context_lock:
            for item in new_items:
                item.reasoning_version = cycle_start_version
            self.last_reasoned_version = cycle_start_version

    def _evaluate_reasoning_completion(self, context_hash_before: str) -> Dict[str, Any]:
        """Evaluate if reasoning cycle is complete and should continue"""
        
        # Compute context hash after reasoning
        context_hash_after = self._compute_context_hash()
        context_changed = context_hash_before != context_hash_after
        
        # Track hash history for oscillation detection
        self.context_hash_history.append(context_hash_after)
        # Keep only recent history
        if len(self.context_hash_history) > self.oscillation_detection_window * 2:
            self.context_hash_history = self.context_hash_history[-self.oscillation_detection_window:]
        
        print(f"   Context hash: {context_hash_before[:8]} -> {context_hash_after[:8]}")
        
        # Check for oscillation patterns
        oscillation = self._detect_oscillation(context_hash_after)
        if oscillation['oscillating']:
            print(f"   OSCILLATION DETECTED: {oscillation['pattern']} pattern")
            print(f"      States: {oscillation['states']}")
            return {
                'should_stop_reasoning': True,
                'stop_reason': f"Oscillation detected: {oscillation['pattern']} pattern",
                'context_changed': True,
                'oscillation': oscillation
            }
        
        # Update sterile cycle tracking
        if context_changed:
            self.consecutive_sterile_cycles = 0
            self.last_context_hash = context_hash_after
            print(f"Reasoning cycle complete - CONTEXT CHANGED (loop {self.reasoning_loop_count})")
        else:
            self.consecutive_sterile_cycles += 1
            print(f"Reasoning cycle complete - no context changes ({self.consecutive_sterile_cycles} sterile)")
            
            # Stop if too many sterile cycles
            if self.consecutive_sterile_cycles >= self.max_sterile_cycles:
                return {
                    'should_stop_reasoning': True,
                    'stop_reason': f'No context changes for {self.max_sterile_cycles} cycles',
                    'context_changed': False
                }
        
        self.stats['reasoning_cycles'] += 1
        
        # Generate situation report
        with self.context_lock:
            new_items = [item for item in self.context_items 
                          if item.reasoning_version == self.context_version]
        situation_report = self._generate_situation_report(new_items)
        print(situation_report)
        
        return {
            'should_stop_reasoning': False,
            'stop_reason': None,
            'context_changed': context_changed
        }

    # ========================================================================
    # LLM INTEGRATION - Hypothesis generation and reasoning
    # ========================================================================

    def _generate_initial_hypotheses(self, evidence_items: List[ContextItem]):
        """Generate initial hypotheses from evidence using LLM"""
        print("   Generating initial hypotheses...")
        
        # Build clean evidence data for LLM
        evidence_summary = "\n".join([item.to_tuple_string() for item in evidence_items])
        
        # Use LLM to generate hypothesis suggestions
        hypothesis_suggestions = self.llm_manager.generate_initial_hypotheses(
            evidence_items=evidence_summary
        )
        
        print(f"   LLM Hypothesis Suggestions: {hypothesis_suggestions[:100]}...")
        
        # Parse and create hypotheses
        self._create_hypotheses_from_suggestions(hypothesis_suggestions)
        
        self.stats['hypotheses_generated'] += 1
    
    def _reason_about_new_items(self, context_summary: str, new_items: List[ContextItem]) -> bool:
        """Reason about new items using LLM - returns True if changes made"""
        print("   Reasoning about new items against existing context...")
        
        # Build clean new items data for LLM
        new_items_summary = "\n".join([item.to_tuple_string() for item in new_items])
        
        # Use LLM to analyze context changes
        reasoning_analysis = self.llm_manager.reason_about_context_changes(
            context_summary=context_summary,
            new_items=new_items_summary
        )
        
        print(f"   LLM Reasoning: {reasoning_analysis[:100]}...")
        
        # Apply reasoning results
        changes_made = self._apply_reasoning_analysis(reasoning_analysis, new_items)
        
        return changes_made

    def _apply_reasoning_analysis(self, analysis: str, new_items: List[ContextItem]) -> bool:
        """Apply LLM reasoning decisions to context"""
        try:
            decisions = json.loads(analysis)
        except json.JSONDecodeError as e:
            print(f"   ERROR: JSON parse failed: {e}")
            return False
        
        changes_made = False
        matched_hypotheses = set()  # Track which hypotheses have been matched
        
        with self.context_lock:
            # Update existing hypotheses based on LLM decisions
            for llm_hyp in decisions.get('hypotheses', []):
                llm_content = llm_hyp.get('content') or llm_hyp.get('statement', '')
                llm_confidence = llm_hyp.get('confidence', 0.6)
                llm_status = llm_hyp.get('status', 'active').lower()
                
                hypothesis_updated = False
                
                for item in self.context_items:
                    if item.item_type != ItemType.HYPOTHESIS:
                        continue
                    
                    # Skip if this hypothesis was already matched
                    if id(item) in matched_hypotheses:
                        continue
                    
                    # Match hypothesis by content similarity using LLM matching
                    if self._hypothesis_matches(item.content, llm_content):
                        print(f"   DUPLICATE ELIMINATED: LLM hypothesis matched existing")
                        print(f"     Eliminated: {llm_content[:80]}...")

                        old_confidence = item.confidence
                        old_status = item.status
                        
                        # Apply LLM's decisions with bounds checking 
                        item.confidence = max(0.0, min(1.0, float(llm_confidence)))

                        if llm_status == 'active':
                            item.status = Status.ACTIVE
                        elif llm_status == 'weakened':
                            item.status = Status.WEAKENED
                        elif llm_status == 'dormant':
                            item.status = Status.DORMANT
                        
                        # Track changes
                        confidence_change = abs(old_confidence - item.confidence)
                        if confidence_change > 0.05:
                            changes_made = True
                        
                        if old_status != item.status:
                            changes_made = True
                            if old_status == Status.DORMANT and item.status == Status.ACTIVE:
                                self._log_revival(item, "LLM revival: Supporting evidence detected")
                        
                        matched_hypotheses.add(id(item))
                        hypothesis_updated = True
                        break
                
                if not hypothesis_updated:
                    print(f"   WARNING: Could not find hypothesis to update: {llm_content[:50]}...")
            
            # Create new hypotheses
            for new_hyp in decisions.get('new_hypotheses', []):
                content = new_hyp.get('content') or new_hyp.get('statement', '')
                confidence = new_hyp.get('confidence', 0.6)
                status_str = new_hyp.get('status', 'active').lower()
                
                if content:
                    hypothesis_item = ContextItem(
                        content=content,
                        timestamp=datetime.now(),
                        item_type=ItemType.HYPOTHESIS,
                        status=Status.ACTIVE if status_str == 'active' else 
                               Status.WEAKENED if status_str == 'weakened' else Status.DORMANT,
                        confidence=float(confidence),
                        importance=0.9,
                        source="llm_reasoning",
                        tags=["llm_generated", "hypothesis"],
                        reasoning_version=self.context_version
                    )
                    
                    self.context_items.append(hypothesis_item)
                    changes_made = True
        
        return changes_made

    def _create_hypotheses_from_suggestions(self, suggestions: str):
        """Create hypothesis items from LLM suggestions with improved parsing"""
        print(f"   Processing LLM Hypothesis Suggestions:")
        print(f"   Raw LLM output: {suggestions}")
        
        hypotheses_created = []
        suggestion_lines = suggestions.split('\n')
        
        for line in suggestion_lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or len(line) < 10:
                continue
            
            # Look for structured format: "HYPOTHESIS: ... | CONFIDENCE: ... | REASONING: ..."
            if 'HYPOTHESIS:' in line.upper() and '|' in line:
                hypothesis_item = self._parse_structured_hypothesis(line)
                if hypothesis_item:
                    hypotheses_created.append(hypothesis_item)
                    continue
            
            # Fallback: Look for hypothesis patterns in unstructured response
            hypothesis_item = self._parse_unstructured_hypothesis(line)
            if hypothesis_item:
                hypotheses_created.append(hypothesis_item)
        
        # Report what was created
        self._report_hypothesis_creation(hypotheses_created, suggestions)

    def _parse_structured_hypothesis(self, line: str) -> Optional[ContextItem]:
        """Parse structured hypothesis format: HYPOTHESIS: ... | CONFIDENCE: ... | REASONING: ..."""
        parts = line.split('|')
        hypothesis_part = None
        confidence_part = None
        
        for part in parts:
            part = part.strip()
            if part.upper().startswith('HYPOTHESIS:'):
                hypothesis_part = part[11:].strip()  # Remove "HYPOTHESIS:"
            elif part.upper().startswith('CONFIDENCE:'):
                confidence_str = part[11:].strip()  # Remove "CONFIDENCE:"
                try:
                    confidence_part = float(confidence_str)
                except ValueError:
                    confidence_part = 0.6  # Default
        
        if hypothesis_part and len(hypothesis_part) > 15:
            confidence = confidence_part if confidence_part else 0.6
            
            # Create the hypothesis
            hypothesis_item = ContextItem(
                content=hypothesis_part,
                timestamp=datetime.now(),
                item_type=ItemType.HYPOTHESIS,
                status=Status.ACTIVE,
                confidence=confidence,
                importance=0.9,
                source="llm_generation",
                tags=["generated", "hypothesis"],
                reasoning_version=self.context_version
            )
            
            with self.context_lock:
                self.context_items.append(hypothesis_item)
            
            return hypothesis_item
        
        return None

    def _parse_unstructured_hypothesis(self, line: str) -> Optional[ContextItem]:
        """Parse unstructured hypothesis from general LLM response"""
        hypothesis_indicators = [
            'hypothesis:', 'theory:', 'conclusion:', 'assessment:', 
            'suggests that', 'indicates that', 'evidence shows',
            'likely that', 'appears that', 'pattern suggests'
        ]
        
        is_hypothesis = False
        hypothesis_content = line
        
        # Check for explicit hypothesis markers
        for indicator in hypothesis_indicators:
            if indicator in line.lower():
                is_hypothesis = True
                # Clean up the content
                if ':' in line:
                    hypothesis_content = line.split(':', 1)[1].strip()
                else:
                    hypothesis_content = line
                break
        
        # Also capture sentences that look like domain hypotheses
        domain_keywords = self.domain_config.hypothesis_keywords
        if (not is_hypothesis and 
            len(line) > 20 and 
            any(keyword in line.lower() for keyword in domain_keywords) and
            ('should' in line.lower() or 'will' in line.lower() or 'is' in line.lower())):
            is_hypothesis = True
            hypothesis_content = line
        
        if is_hypothesis and len(hypothesis_content) > 15:
            # Clean up common prefixes
            prefixes_to_remove = ['- ', '\u2022 ', '1. ', '2. ', '3. ', '4. ', '5. ']
            for prefix in prefixes_to_remove:
                if hypothesis_content.startswith(prefix):
                    hypothesis_content = hypothesis_content[len(prefix):].strip()
    
            # Determine confidence based on language used
            confidence = 0.6  # Default
            if any(word in hypothesis_content.lower() for word in ['strong', 'clear', 'evident', 'definite']):
                confidence = 0.8
            elif any(word in hypothesis_content.lower() for word in ['likely', 'probable', 'suggests']):
                confidence = 0.7
            elif any(word in hypothesis_content.lower() for word in ['possible', 'may', 'might', 'could']):
                confidence = 0.5
            
            # Create the hypothesis
            hypothesis_item = ContextItem(
                content=hypothesis_content,
                timestamp=datetime.now(),
                item_type=ItemType.HYPOTHESIS,
                status=Status.ACTIVE,
                confidence=confidence,
                importance=0.9,
                source="llm_generation_fallback",
                tags=["generated", "hypothesis"],
                reasoning_version=self.context_version
            )
            
            with self.context_lock:
                self.context_items.append(hypothesis_item)
            
            return hypothesis_item
        
        return None

    def _report_hypothesis_creation(self, hypotheses_created: List[ContextItem], suggestions: str):
        """Report on hypothesis creation results"""
        if hypotheses_created:
            print(f"   Created {len(hypotheses_created)} hypotheses from LLM suggestions:")
            for i, hyp in enumerate(hypotheses_created, 1):
                print(f"   {i}. {hyp.content}")
                print(f"      Confidence: {hyp.confidence:.2f}")
        else:
            print(f"   No clear hypotheses found in LLM suggestions")
            print(f"   LLM response may need better prompting for structured hypothesis generation")
            
            # Create a fallback hypothesis based on the general suggestion
            if len(suggestions) > 20:
                fallback_hypothesis = ContextItem(
                    content=f"Analysis suggests: {suggestions[:200]}...",
                    timestamp=datetime.now(),
                    item_type=ItemType.HYPOTHESIS,
                    status=Status.ACTIVE,
                    confidence=0.5,
                    importance=0.7,
                    source="llm_generation_fallback",
                    tags=["generated", "hypothesis", "fallback"],
                    reasoning_version=self.context_version
                )
                
                with self.context_lock:
                    self.context_items.append(fallback_hypothesis)
                    
                print(f"   Created fallback hypothesis: {fallback_hypothesis.content}")

    def _hypothesis_matches(self, existing_content: str, llm_content: str) -> bool:
        """Check if LLM hypothesis content matches an existing hypothesis using LLM semantic matching"""
        if not existing_content or not llm_content:
            return False
        
        # Use LLM for semantic matching
        return self.llm_manager.check_hypothesis_equivalence(existing_content, llm_content)

    # ========================================================================
    # SAFETY MECHANISMS - Oscillation detection and progress tracking
    # ========================================================================

    def _detect_oscillation(self, current_hash: str) -> Dict[str, Any]:
        """Detect if reasoning is oscillating between states"""
        if len(self.context_hash_history) < self.oscillation_detection_window:
            return {'oscillating': False, 'pattern': None}
        
        recent_hashes = self.context_hash_history[-self.oscillation_detection_window:]
        
        # Check for simple 2-state oscillation (A-B-A-B-A-B)
        if len(set(recent_hashes)) == 2:
            hash_a, hash_b = list(set(recent_hashes))
            expected_pattern = [hash_a, hash_b] * (self.oscillation_detection_window // 2)
            if recent_hashes == expected_pattern or recent_hashes == expected_pattern[1:] + [expected_pattern[0]]:
                return {
                    'oscillating': True,
                    'pattern': '2-state',
                    'states': [hash_a[:8], hash_b[:8]]
                }
        
        # Check for 3-state cycles (A-B-C-A-B-C)
        if len(set(recent_hashes)) == 3 and len(recent_hashes) >= 6:
            if (recent_hashes[0] == recent_hashes[3] and 
                recent_hashes[1] == recent_hashes[4] and 
                recent_hashes[2] == recent_hashes[5]):
                return {
                    'oscillating': True,
                    'pattern': '3-state',
                    'states': list(set(recent_hashes))
                }
        
        return {'oscillating': False, 'pattern': None}

    def _compute_context_hash(self) -> str:
        """Compute hash of entire context state for oscillation detection"""
        
        # Create deterministic representation of context
        context_data = []
        
        for item in sorted(self.context_items, key=lambda x: x.timestamp):
            item_repr = (
                item.content,
                item.item_type.value,
                item.status.value,
                round(item.confidence, 3),  # Round to avoid floating point noise
                round(item.importance, 3),
                item.source,
                tuple(sorted(item.tags))
            )
            context_data.append(item_repr)
        
        # Convert to string and hash
        context_str = str(context_data)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]  # First 16 chars

    def _track_reasoning_changes(self, before_items: List[ContextItem]) -> Dict[str, Any]:
        """Track what changed during reasoning cycle for progress reporting"""
        changes = {
            'confidence_updates': [],
            'status_changes': [],
            'revivals': [],
            'new_hypotheses': [],
            'dormant_hypotheses': [],
            'deleted_hypotheses': []
        }
        
        # Create lookup for before state
        before_lookup = {item.content: item for item in before_items}
        
        with self.context_lock:
            current_hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
            
            for current_item in current_hypotheses:
                if current_item.content in before_lookup:
                    before_item = before_lookup[current_item.content]
                    
                    # Track confidence changes
                    conf_change = current_item.confidence - before_item.confidence
                    if abs(conf_change) > 0.05:  # Significant change threshold
                        changes['confidence_updates'].append({
                            'hypothesis': current_item.content[:60] + "..." if len(current_item.content) > 60 else current_item.content,
                            'old_confidence': before_item.confidence,
                            'new_confidence': current_item.confidence,
                            'change': conf_change,
                            'direction': 'UP' if conf_change > 0 else 'DOWN'
                        })
                    
                    # Track status changes
                    if current_item.status != before_item.status:
                        changes['status_changes'].append({
                            'hypothesis': current_item.content[:60] + "..." if len(current_item.content) > 60 else current_item.content,
                            'old_status': before_item.status.value,
                            'new_status': current_item.status.value,
                            'confidence': current_item.confidence
                        })
                        
                        # Special tracking for revivals and dormancy
                        if before_item.status == Status.DORMANT and current_item.status == Status.ACTIVE:
                            changes['revivals'].append(current_item.content[:60] + "...")
                        elif current_item.status == Status.DORMANT:
                            changes['dormant_hypotheses'].append(current_item.content[:60] + "...")
                else:
                    # New hypothesis created
                    changes['new_hypotheses'].append({
                        'hypothesis': current_item.content[:60] + "..." if len(current_item.content) > 60 else current_item.content,
                        'confidence': current_item.confidence,
                        'source': current_item.source
                    })
            
            # Check for deleted hypotheses
            current_contents = {item.content for item in current_hypotheses}
            for before_content, before_item in before_lookup.items():
                if (before_item.item_type == ItemType.HYPOTHESIS and 
                    before_content not in current_contents):
                    changes['deleted_hypotheses'].append(before_content[:60] + "...")
        
        return changes

    def _report_reasoning_progress(self, cycle_number: int, changes: Dict[str, Any]):
        """Report what happened during this reasoning cycle"""
        print(f"\n   CYCLE {cycle_number} PROGRESS REPORT:")
        
        total_changes = (len(changes['confidence_updates']) + 
                        len(changes['status_changes']) + 
                        len(changes['new_hypotheses']) + 
                        len(changes['deleted_hypotheses']))
        
        if total_changes == 0:
            print(f"   No significant changes this cycle")
            return
        
        # Report confidence updates
        if changes['confidence_updates']:
            print(f"   CONFIDENCE UPDATES ({len(changes['confidence_updates'])}):")
            for update in changes['confidence_updates']:
                print(f"      {update['direction']} {update['hypothesis']}")
                print(f"         {update['old_confidence']:.2f} -> {update['new_confidence']:.2f} (change: {update['change']:+.2f})")
        
        # Report status changes
        if changes['status_changes']:
            print(f"   STATUS CHANGES ({len(changes['status_changes'])}):")
            for change in changes['status_changes']:
                old_emoji = 'ACTIVE' if change['old_status'] == 'active' else 'WEAKENED' if change['old_status'] == 'weakened' else 'DORMANT'
                new_emoji = 'ACTIVE' if change['new_status'] == 'active' else 'WEAKENED' if change['new_status'] == 'weakened' else 'DORMANT'
                print(f"      {old_emoji}->{new_emoji} {change['hypothesis']}")
                print(f"         {change['old_status']} -> {change['new_status']} (conf: {change['confidence']:.2f})")
        
        # Report new hypotheses
        if changes['new_hypotheses']:
            print(f"   NEW HYPOTHESES ({len(changes['new_hypotheses'])}):")
            for new_hyp in changes['new_hypotheses']:
                print(f"      NEW {new_hyp['hypothesis']}")
                print(f"         Initial confidence: {new_hyp['confidence']:.2f} (from {new_hyp['source']})")
        
        # Report deletions
        if changes['deleted_hypotheses']:
            print(f"   DELETED HYPOTHESES ({len(changes['deleted_hypotheses'])}):")
            for deleted in changes['deleted_hypotheses']:
                print(f"      DELETED {deleted}")
        
        # Special callouts
        if changes['revivals']:
            print(f"   REVIVALS ({len(changes['revivals'])}):")
            for revival in changes['revivals']:
                print(f"      REVIVED {revival}")

    def _log_revival(self, hypothesis: ContextItem, reason: str):
        """Log hypothesis revival for tracking"""
        revival_item = ContextItem(
            content=f"REVIVAL: {hypothesis.content[:50]}... - {reason}",
            timestamp=datetime.now(),
            item_type=ItemType.REVIVAL,
            status=Status.ACTIVE,
            confidence=0.8,
            importance=0.9,
            source="reasoning_cycle",
            tags=["revival"],
            reasoning_version=self.context_version
        )
        
        self.context_items.append(revival_item)
        self.stats['revivals_detected'] += 1
        print(f"   REVIVAL: {hypothesis.content[:50]}...")

    # ========================================================================
    # PROGRESS REPORTING & SUMMARIES
    # ========================================================================

    def _generate_situation_report(self, new_items: List[ContextItem]) -> str:
        """Generate detective-style 'where are we' summary"""
        
        with self.context_lock:
            # Get current state
            active_hyp = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS and i.status == Status.ACTIVE]
            weakened_hyp = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS and i.status == Status.WEAKENED]
            dormant_hyp = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS and i.status == Status.DORMANT]
            recent_evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE and 
                             (datetime.now() - i.timestamp).total_seconds() < 3600]  # Last hour
            revivals = [i for i in self.context_items if i.item_type == ItemType.REVIVAL]
        
        # Build detective-style report
        report = f"\n{'='*60}\n"
        report += f"SITUATION REPORT - REASONING CYCLE {self.reasoning_loop_count}\n"
        report += f"{'='*60}\n"
        
        # Current working theories
        if active_hyp:
            report += f"\nACTIVE THEORIES ({len(active_hyp)}):\n"
            for i, hyp in enumerate(sorted(active_hyp, key=lambda x: x.confidence, reverse=True), 1):
                confidence_trend = "UP" if hyp.confidence > 0.7 else "STABLE" if hyp.confidence > 0.5 else "DOWN"
                report += f"   {i}. {confidence_trend} {hyp.content}\n"
                report += f"      Confidence: {hyp.confidence:.2f}\n"
        
        # What just happened (new evidence impact)
        if new_items:
            evidence_items = [item for item in new_items if item.item_type == ItemType.EVIDENCE]
            if evidence_items:
                report += f"\nNEW DEVELOPMENTS ({len(evidence_items)} items):\n"
                for i, evidence in enumerate(evidence_items, 1):
                    age_minutes = (datetime.now() - evidence.timestamp).total_seconds() / 60
                    report += f"   {i}. {evidence.content}\n"
                    report += f"      Source: {evidence.source}, {age_minutes:.0f} minutes ago\n"
                
                # How new evidence affects theories
                report += f"\nIMPACT ASSESSMENT:\n"
                impact_summary = self._assess_evidence_impact(evidence_items, active_hyp)
                report += impact_summary
        
        # Case status changes
        if weakened_hyp or dormant_hyp or revivals:
            report += f"\nSTATUS CHANGES:\n"
            
            if revivals:
                report += f"   REVIVED THEORIES: {len(revivals)} dormant theories brought back into consideration\n"
            
            if weakened_hyp:
                report += f"   WEAKENED THEORIES ({len(weakened_hyp)}):\n"
                for i, hyp in enumerate(weakened_hyp, 1):
                    report += f"      {i}. {hyp.content}\n"
                    report += f"         Confidence: {hyp.confidence:.2f}\n"
            
            if dormant_hyp:
                report += f"   DORMANT THEORIES: {len(dormant_hyp)} theories shelved (still recoverable)\n"
        
        # Current focus and next steps
        report += f"\nCURRENT FOCUS:\n"
        if active_hyp:
            top_theory = max(active_hyp, key=lambda x: x.confidence)
            report += f"   Thesis with highest confidence:\n"
            report += f"     {top_theory.content}\n"
            report += f"   Confidence level: {top_theory.confidence:.2f} ({self._confidence_description(top_theory.confidence)})\n"
        
        # Evidence strength assessment
        if recent_evidence:
            strong_evidence = len([e for e in recent_evidence if e.confidence > 0.8])
            total_evidence = len(recent_evidence)
            report += f"   Recent evidence quality: {strong_evidence}/{total_evidence} high-confidence items\n"
        
        # What we're watching for
        report += f"\nMONITORING FOR:\n"
        if active_hyp:
            report += f"   Additional evidence to strengthen primary theory\n"
            report += f"   Contradictory signals that might challenge assumptions\n"
        if dormant_hyp:
            report += f"   New developments that might revive {len(dormant_hyp)} dormant theories\n"
        
        report += f"\n{'='*60}\n"
        
        return report

    def _assess_evidence_impact(self, evidence_items: List[ContextItem], hypotheses: List[ContextItem]) -> str:
        """Assess how new evidence impacts current theories"""
        if not evidence_items or not hypotheses:
            return "   No significant impact assessment available\n"
        
        impact = ""
        
        # Simplified impact analysis (in production, could use LLM for this too)
        for evidence in evidence_items:
            evidence_words = set(evidence.content.lower().split())
            
            for hyp in hypotheses:
                hyp_words = set(hyp.content.lower().split())
                overlap = len(evidence_words.intersection(hyp_words))
                
                if overlap > 2:  # Simple relevance check
                    if evidence.confidence > 0.8:
                        if any(word in evidence.content.lower() for word in ['exceeded', 'success', 'improved', 'strong']):
                            impact += f"   Evidence SUPPORTS: {hyp.content[:45]}...\n"
                        elif any(word in evidence.content.lower() for word in ['failed', 'declined', 'issues', 'problems']):
                            impact += f"   Evidence CHALLENGES: {hyp.content[:45]}...\n"
                        else:
                            impact += f"   Evidence INFORMS: {hyp.content[:45]}...\n"
        
        if not impact:
            impact = "   Evidence provides general context, no direct theory impact\n"
        
        return impact

    def _confidence_description(self, confidence: float) -> str:
        """Convert confidence score to descriptive text"""
        if confidence >= 0.9:
            return "very high confidence"
        elif confidence >= 0.8:
            return "high confidence"
        elif confidence >= 0.7:
            return "good confidence"
        elif confidence >= 0.6:
            return "moderate confidence"
        elif confidence >= 0.5:
            return "low confidence"
        else:
            return "very low confidence"

    def _generate_deep_thought_summary(self):
        """Generate comprehensive summary of what was accomplished during deep thought session"""
        
        if not self.deep_thought_start:
            return
        
        session_duration = (datetime.now() - self.deep_thought_start).total_seconds()
        
        with self.context_lock:
            # Get current state
            active_hyp = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS and i.status == Status.ACTIVE]
            weakened_hyp = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS and i.status == Status.WEAKENED]
            dormant_hyp = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS and i.status == Status.DORMANT]
            all_hyp = active_hyp + weakened_hyp + dormant_hyp
            revivals = [i for i in self.context_items if i.item_type == ItemType.REVIVAL]
            recent_evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE and 
                             (datetime.now() - i.timestamp).total_seconds() < session_duration + 60]
        
        print(f"\n{'='*70}")
        print(f"DEEP THOUGHT SESSION COMPLETE")
        print(f"{'='*70}")
        print(f"Session Duration: {session_duration/60:.1f} minutes")
        print(f"Reasoning Cycles: {self.reasoning_loop_count}")
        print(f"Loop Counter: Used {self.reasoning_loop_count} of {self.max_reasoning_loops} max loops")
        
        # DIAGNOSTIC: Why no strong conclusions?
        if not active_hyp and len(all_hyp) == 0:
            print(f"\nDIAGNOSTIC ALERT: NO HYPOTHESES EXIST")
            print(f"   Expected: LLM should generate hypotheses from evidence")
            print(f"   Reality: No hypotheses created during {self.reasoning_loop_count} cycles")
            print(f"   Possible causes:")
            print(f"     - LLM hypothesis generation not working")
            print(f"     - Insufficient evidence threshold")
            print(f"     - LLM reasoning module failure")
        
        elif not active_hyp and all_hyp:
            print(f"\nDIAGNOSTIC ALERT: ALL HYPOTHESES WEAKENED/DORMANT")
            print(f"   Total hypotheses created: {len(all_hyp)}")
            print(f"   Active: {len(active_hyp)}, Weakened: {len(weakened_hyp)}, Dormant: {len(dormant_hyp)}")
            print(f"   This suggests reasoning is too aggressive in lowering confidence")
            
            print(f"\n   All Hypotheses Status:")
            for i, hyp in enumerate(all_hyp, 1):
                print(f"   {i}. {hyp.content}")
                print(f"      Status: {hyp.status.value}, Confidence: {hyp.confidence:.2f}")
        
        # DIAGNOSTIC: Show actual hypothesis content vs placeholder
        if active_hyp:
            print(f"\nFINAL ASSESSMENT:")
            top_theory = max(active_hyp, key=lambda x: x.confidence)
            
            # Check if this is a placeholder hypothesis
            if "LLM-generated hypothesis based on evidence patterns" in top_theory.content:
                print(f"   PLACEHOLDER HYPOTHESIS DETECTED:")
                print(f"   The system created a generic placeholder instead of real reasoning")
                print(f"   This indicates LLM hypothesis generation is not working properly")
                print(f"   Content: {top_theory.content}")
                print(f"   Source: {top_theory.source}")
                print(f"   Need to implement proper LLM hypothesis generation from suggestions")
            else:
                print(f"   Primary Conclusion: {top_theory.content}")
            
            print(f"   Confidence Level: {top_theory.confidence:.2f} ({self._confidence_description(top_theory.confidence)})")
            
            if len(active_hyp) > 1:
                other_theories = [h for h in active_hyp if h != top_theory]
                print(f"\n   Secondary Theories ({len(other_theories)}):")
                for i, theory in enumerate(other_theories, 1):
                    if "LLM-generated hypothesis based on evidence patterns" in theory.content:
                        print(f"   {i}. PLACEHOLDER: {theory.content}")
                    else:
                        print(f"   {i}. {theory.content}")
                    print(f"      Confidence: {theory.confidence:.2f}")
        
        # What changed during session
        if revivals:
            print(f"\nSESSION INSIGHTS:")
            print(f"   Theories Revived ({len(revivals)}):")
            for i, revival in enumerate(revivals, 1):
                print(f"   {i}. {revival.content}")
        
        if dormant_hyp:
            print(f"\n   Theories Set Aside ({len(dormant_hyp)}):")
            for i, dormant in enumerate(dormant_hyp, 1):
                print(f"   {i}. {dormant.content}")
                print(f"      (confidence dropped to {dormant.confidence:.2f})")
        
        # Evidence considered
        if recent_evidence:
            evidence_sources = list(set(e.source for e in recent_evidence))
            print(f"\nEVIDENCE ANALYZED:")
            print(f"   {len(recent_evidence)} pieces of evidence from {len(evidence_sources)} sources")
            print(f"   Key sources: {', '.join(evidence_sources[:3])}")
            
            high_confidence_evidence = [e for e in recent_evidence if e.confidence > 0.8]
            if high_confidence_evidence:
                print(f"   {len(high_confidence_evidence)} high-confidence items shaped conclusions")
                
                # DIAGNOSTIC: Show what evidence was available
                if not active_hyp:
                    print(f"\n   Evidence Available for Hypothesis Generation:")
                    for i, evidence in enumerate(recent_evidence, 1):
                        print(f"   {i}. {evidence.content}")
                        print(f"      Source: {evidence.source}, Confidence: {evidence.confidence:.2f}")
        
        # Current readiness state
        print(f"\nCURRENT STATUS:")
        if active_hyp:
            avg_confidence = sum(h.confidence for h in active_hyp) / len(active_hyp)
            print(f"   Working theories: {len(active_hyp)} active (avg confidence: {avg_confidence:.2f})")
        else:
            print(f"   No strong theories currently - waiting for more evidence")
            if recent_evidence and len(recent_evidence) >= 3:
                print(f"   ALERT: {len(recent_evidence)} pieces of evidence available but no theories generated")
                print(f"   Suggests LLM reasoning issue - may need investigation")
        
        print(f"   System ready for new assertions and analysis")
        
        print(f"\n{'='*70}\n")

    def _display_full_context_window(self):
        """Display the complete context window for debugging"""
        print(f"\n{'='*80}")
        print(f"FULL CONTEXT WINDOW DUMP")
        print(f"{'='*80}")
        print(f"Total items: {len(self.context_items)}")
        print(f"Context version: {self.context_version}")
        print(f"Estimated tokens: {sum(int(item.token_estimate()) for item in self.context_items)}")
        
        with self.context_lock:
            for i, item in enumerate(self.context_items, 1):
                age_minutes = (datetime.now() - item.timestamp).total_seconds() / 60
                print(f"\n[{i}] {item.item_type.value.upper()} | {item.status.value} | conf:{item.confidence:.2f}")
                print(f"    Content: {item.content}")
                print(f"    Source: {item.source} | {age_minutes:.1f}min ago | v{item.reasoning_version}")
                print(f"    Tags: {', '.join(item.tags)}")
        
        print(f"{'='*80}\n")

    # ========================================================================
    # CONTEXT MANAGEMENT - Compression and utility functions
    # ========================================================================

    def _perform_compression(self):
        """Perform context compression to manage token limits"""
        print("   Performing context compression...")
        
        # Simplified compression (keep important items)
        with self.context_lock:
            important_items = [
                item for item in self.context_items
                if (item.item_type in [ItemType.HYPOTHESIS, ItemType.REVIVAL] or
                    item.importance > 0.8 or
                    (datetime.now() - item.timestamp).total_seconds() < 3600)
            ]
            
            old_count = len(self.context_items)
            self.context_items = important_items
            
            print(f"   Compressed: {old_count} -> {len(self.context_items)} items")
            self.stats['compressions_performed'] += 1

    def _build_context_summary(self) -> str:
        """Build context summary for LLM consumption"""
        hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
        evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE]
        
        summary = "CURRENT REASONING CONTEXT:\n\n"
        
        if hypotheses:
            summary += "HYPOTHESES:\n"
            for h in hypotheses:
                summary += f"  {h.to_tuple_string()}\n"
            summary += "\n"
        
        if evidence:
            summary += "EVIDENCE:\n"
            for e in evidence[-10:]:  # Recent evidence
                summary += f"  {e.to_tuple_string()}\n"
            summary += "\n"
        
        return summary

    def _assess_importance(self, content: str) -> float:
        """Quick importance assessment based on content"""
        importance = 0.5
        high_impact_words = ['critical', 'significant', 'major', 'exceeded', 'failed']
        if any(word in content.lower() for word in high_impact_words):
            importance += 0.3
        return min(1.0, importance)

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content using domain-specific keywords"""
        content_lower = content.lower()
        tags = []
        for tag, keywords in self.domain_config.keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        return tags or ['general']

    def _capture_investigation_results(self) -> Dict[str, Any]:
        """Capture current investigation state as results"""
        with self.context_lock:
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
            evidence = [i for i in self.context_items if i.item_type == ItemType.EVIDENCE]
            
            # Build results summary
            results = {
                "timestamp": datetime.now().isoformat(),
                "total_items": len(self.context_items),
                "evidence_count": len(evidence),
                "hypotheses_count": len(hypotheses),
                "reasoning_cycles": self.stats['reasoning_cycles'],
                "hypotheses": [
                    {
                        "content": h.content,
                        "confidence": h.confidence,
                        "status": h.status.value,
                        "source": h.source
                    } for h in hypotheses
                ],
                "evidence_summary": [
                    {
                        "content": e.content[:100] + "..." if len(e.content) > 100 else e.content,
                        "source": e.source,
                        "confidence": e.confidence
                    } for e in evidence
                ],
                "stats": self.stats.copy()
            }
            
            return results

    def _store_investigation_results(self, investigation_id: str, results: Dict[str, Any]):
        """Store investigation results (placeholder for SQLite integration)"""
        # TODO: Store in SQLite database with TTL
        print(f"STUB: Storing results for investigation {investigation_id}")
        print(f"   Hypotheses: {results['hypotheses_count']}")
        print(f"   Evidence: {results['evidence_count']}")
        print(f"   Reasoning cycles: {results['reasoning_cycles']}")
        if results['hypotheses']:
            top_hypothesis = max(results['hypotheses'], key=lambda x: x['confidence'])
            print(f"   Top hypothesis: {top_hypothesis['content'][:60]}... (confidence: {top_hypothesis['confidence']:.2f})")

# ============================================================================
# CSV EVIDENCE PROCESSING
# ============================================================================

def parse_evidence_csv(filename: str) -> List[Dict[str, Any]]:
    """Parse CSV evidence file into structured commands"""
    commands = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 2):
                try:
                    cmd_type = row['type'].strip().lower()
                    
                    if cmd_type == 'clear':
                        investigation_name = row.get('investigation_name', '').strip() or "csv_investigation"
                        commands.append({
                            'type': 'clear',
                            'investigation_name': investigation_name
                        })
                    
                    elif cmd_type == 'evidence':
                        content = row['content'].strip()
                        source = row['source'].strip()
                        confidence_str = row.get('confidence', '0.85').strip()
                        confidence = float(confidence_str) if confidence_str else 0.85
                        
                        if content and source:
                            commands.append({
                                'type': 'evidence',
                                'content': content,
                                'source': source,
                                'confidence': confidence
                            })
                    
                    elif cmd_type == 'delay':
                        delay_str = row.get('delay_seconds', '1.0').strip()
                        delay_seconds = float(delay_str) if delay_str else 1.0
                        commands.append({
                            'type': 'delay',
                            'seconds': delay_seconds
                        })
                    
                    elif cmd_type == 'query':
                        query_text = row['content'].strip()
                        if query_text:
                            commands.append({
                                'type': 'query',
                                'text': query_text
                            })
                
                except ValueError as e:
                    print(f"Warning: Row {row_num} has invalid number, using defaults")
                except Exception as e:
                    print(f"Error parsing row {row_num}: {e}")
                    continue
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Evidence file '{filename}' not found")
    
    return commands

def execute_evidence_csv(engine: AsyncReasoningEngine, filename: str):
    """Execute commands from CSV evidence file"""
    commands = parse_evidence_csv(filename)
    
    print(f"Loaded {len(commands)} commands from {filename}")
    print("="*60)
    
    for i, cmd in enumerate(commands, 1):
        cmd_type = cmd['type']
        print(f"[{i}/{len(commands)}] Executing: {cmd_type.upper()}")
        
        if cmd_type == 'clear':
            investigation_id = engine.clear_context(cmd['investigation_name'])
            print(f"   Cleared context: {investigation_id}")
            
        elif cmd_type == 'evidence':
            engine.add_evidence(cmd['content'], cmd['source'], cmd['confidence'])
            print(f"   Added evidence: {cmd['content'][:50]}...")
            
        elif cmd_type == 'hypothesis':
            engine.add_hypothesis(cmd['content'], cmd['confidence'])
            print(f"   Added hypothesis: {cmd['content'][:50]}...")
            
        elif cmd_type == 'delay':
            print(f"   Waiting {cmd['seconds']} seconds...")
            time.sleep(cmd['seconds'])
            
        elif cmd_type == 'query':
            # Wait for reasoning to complete before querying
            print(f"   Querying: {cmd['text']}")
            print("   Waiting for reasoning to complete before query...")
            while engine.deep_thought_mode or not engine.queued_assertions.empty():
                time.sleep(0.1)
                print(".", end="", flush=True)
            print()  # New line after dots
            
            response = engine.query_context_sync(cmd['text'], timeout=30.0)
            print(f"   Response: {response}")
            
        else:
            print(f"   Warning: Unknown command type '{cmd_type}' - skipping")
        
        time.sleep(0.5)
    
    print("="*60)
    print("CSV execution complete")

# ============================================================================
# DEMO & TESTING FUNCTIONS
# ============================================================================

def demo_async_reasoning():
    """Demonstrate async reasoning engine capabilities"""
    
    print("ASYNCHRONOUS CONTEXT-WATCHING REASONING ENGINE")
    print("=" * 70)
    
    try:
        # Initialize and start
        # Business reasoning (default)
        engine = AsyncReasoningEngine(domain="business", max_context_tokens=3500)
        # Criminal investigation
        #engine = AsyncReasoningEngine(domain="criminal_investigation", max_context_tokens=3500)
        
        engine.start()
        
        # Show initial status
        status = engine.get_status_snapshot()
        print(f"Initial Status: Engine running, waiting for assertions...")
        
        print("\nTESTING ASYNC ASSERTION PROCESSING:")
        
        # Start with clear to establish investigation
        initial_investigation_id = engine.clear_context("demo_investigation")
        print(f"Started investigation: {initial_investigation_id}")
        
        # Add some initial evidence 
        evidence_items = [
            ("Enterprise Q4 sales exceeded targets by 30%", "sales_system"),
            ("SMB customer acquisition costs increased 50%", "sales_analytics"),
            ("Server response times degraded during peak load", "monitoring"),
            ("Database optimization improved query performance by 60%", "engineering"),
        ]
        
        print("\n1. ADDING INITIAL EVIDENCE:")
        for content, source in evidence_items:
            start_time = time.time()
            op_id = engine.add_evidence(content, source, confidence=0.85)
            duration = time.time() - start_time
            time.sleep(0.5)  # Brief pause between additions
        
        print("\n   Initial evidence added - monitoring loop will trigger reasoning when ready...")

        print("\n2. Testing queries:")

        # Add more evidence and let reasoning happen
        print("\n" + "="*70)
        print("ADDING MORE EVIDENCE")
        print("="*70)
        
        # Add additional evidence
        additional_evidence = [
            ("Critical infrastructure failure detected", "alerts"),
            ("Enterprise customers reporting satisfaction improvements", "customer_success"),
            ("New competitor entered SMB market", "market_intelligence"),
            ("System performance metrics normalized after fixes", "monitoring")
        ]
        
        for content, source in additional_evidence:
            engine.add_evidence(content, source, confidence=0.88)
            time.sleep(1)
               
        print("\nWaiting for natural completion...")

        # Wait for both conditions:
        # 1. Deep thought completes 
        # 2. Operation queue is empty
        while engine.deep_thought_mode or not engine.queued_assertions.empty():
            time.sleep(0.1)
            if engine.deep_thought_mode:
                print(".", end="", flush=True)  # Show deep thought progress
            elif not engine.queued_assertions.empty():
                print("q", end="", flush=True)  # Show queue processing

        # Final query
        print("\n4. Final assessment:")
        final_query = "Based on all evidence, what are your key conclusions?"
        response = engine.query_context_sync(final_query, timeout=20.0)
        print(f"\nFinal Assessment: {response}")
              
        print("\nNATURAL COMPLETION ACHIEVED")
        print("- Deep thought: Complete")
        print("- Queue: Empty") 
        print("- System: Idle")

        print("\nDEMONSTRATION COMPLETE")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'engine' in locals():
            try:
                engine.stop()
                print("Reasoning engine stopped")
                
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

def production_reasoning_server():
    """Example of how to run this in production - continuous operation"""
    
    print("PRODUCTION REASONING SERVER")
    print("=" * 50)
    print("Starting reasoning engine in production mode...")
    print("Engine will run continuously until stopped.")
    print("Use Ctrl+C to shutdown gracefully.")
    print("\n" + "="*50)
    
    try:
        # Business reasoning (default)
        engine = AsyncReasoningEngine(domain="business", max_context_tokens=5000)
        # Criminal investigation
        #engine = AsyncReasoningEngine(domain="criminal_investigation", max_context_tokens=5000)
        
        engine.start()
        
        print("Reasoning engine started - waiting for assertions")
        print("\nAPI endpoints available:")
        print("  engine.add_evidence(content, source, confidence)")
        print("  engine.add_hypothesis(content, confidence)")
        print("  investigation_id = engine.clear_context(base_name)")
        print("  results = engine.get_investigation_results(investigation_id)")
        print("  engine.query_context_sync(query, timeout)")
        print("  engine.force_stop_reasoning(reason)")
        print("  engine.get_status_snapshot()")
        print("  engine.toggle_full_context_display(show)")
        print("  engine.configure_max_loops(max_loops)")
        
        # Production servers would integrate with web APIs, message queues, etc.
        # For demo, just keep running
        cycle_count = 0
        while engine.running:
            time.sleep(5)
            cycle_count += 1
            
            # Periodic status log (every 30 seconds)
            if cycle_count % 6 == 0:
                status = engine.get_status_snapshot()
                print(f"Status: {status['total_items']} items, "
                      f"{status['hypotheses']} hypotheses, "
                      f"{'deep thought' if status['deep_thought_mode'] else 'waiting'}, "
                      f"loop {status['reasoning_loop_count']}/{status['max_reasoning_loops']}")
    
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'engine' in locals():
            try:
                engine.stop()
                print("Reasoning engine stopped")
            except Exception as cleanup_error:
                print(f"Error during engine shutdown: {cleanup_error}")

def csv_driven_reasoning(csv_file: str, domain: str = "business"):
    """Run reasoning engine driven by CSV evidence file"""
    
    print(f"CSV-DRIVEN REASONING ENGINE")
    print(f"Evidence file: {csv_file}")
    print(f"Domain: {domain}")
    print("=" * 70)
    
    try:
        # Initialize and start engine
        engine = AsyncReasoningEngine(max_context_tokens=3500, domain=domain)
        engine.start()
        
        # Execute CSV file
        execute_evidence_csv(engine, csv_file)
        
        # Wait for any remaining processing
        print("\nWaiting for reasoning to complete...")
        while engine.deep_thought_mode or not engine.queued_assertions.empty():
            time.sleep(0.1)
            if engine.deep_thought_mode:
                print(".", end="", flush=True)
        
        print(f"\nReasoning complete!")
        
        # Final status
        status = engine.get_status_snapshot()
        print(f"\nFinal Status:")
        print(f"   Total items: {status['total_items']}")
        print(f"   Hypotheses: {status['hypotheses']}")
        print(f"   Reasoning cycles: {status['stats']['reasoning_cycles']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'engine' in locals():
            engine.stop()
            print("Engine stopped")

if __name__ == "__main__":
    import sys
    
    # Command line usage: python -u reasoning.py evidence.csv [domain]
    if len(sys.argv) >= 2:
        csv_file = sys.argv[1]
        domain = sys.argv[2] if len(sys.argv) > 2 else "business"
        csv_driven_reasoning(csv_file, domain)
    else:
        print("Usage: python reasoning.py <evidence.csv> [domain]")
        print("Example: python reasoning.py evidence_business.csv business")

#    demo_async_reasoning()
