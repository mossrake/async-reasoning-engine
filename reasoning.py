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
Reasoning engine for decision support that tracks multiple contradictory hypotheses over noisy/asynchronous business data, 
reviving dormant theories given new evidence.

This code implements a sophisticated reasoning system with:
- Complex templated prompts
- Self-modifying context
- Oscillation detection
- Anti-rumination logic

Architecture:
- Fast assertion queue that only updates context
- Separate reasoning process that watches context changes
- Reasoning involves rewriting the context, and adding to it
- Reasoning cycles run asynchronously without blocking assertions
- Context monitor triggers reasoning when changes detected
- Detailed progress reporting during deep thought cycles
- Issue: DSPy instructions not supported by Azure Open AI; using complex prompts as needed
- Simple loop counter failsafe to prevent infinite reasoning / ruminating / oscillation
"""

import os
import threading
import queue
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import dspy

# Core data structures
class ItemType(Enum):
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    PATTERN = "pattern"
    REVIVAL = "revival"
    SUMMARY = "summary"

class Status(Enum):
    ACTIVE = "active"
    WEAKENED = "weakened" 
    DORMANT = "dormant"

class OperationType(Enum):
    ADD_EVIDENCE = "add_evidence"
    ADD_HYPOTHESIS = "add_hypothesis"
    COMPRESS_CONTEXT = "compress_context"
    QUERY_CONTEXT = "query_context"
    CLEAR_CONTEXT = "clear_context"
    SHUTDOWN = "shutdown"

@dataclass
class ContextItem:
    content: str
    timestamp: datetime
    item_type: ItemType
    status: Status
    confidence: float
    importance: float
    source: str
    tags: List[str]
    access_count: int = 0
    reasoning_version: int = 0  # Track which reasoning cycle processed this
    
    def to_tuple_string(self) -> str:
        """Convert to tuple format for LLM processing"""
        age_minutes = (datetime.now() - self.timestamp).total_seconds() / 60
        return (f"({self.content}, {self.timestamp.strftime('%H:%M')}, "
                f"conf:{self.confidence:.2f}, {self.status.value}, "
                f"{age_minutes:.0f}min_ago, imp:{self.importance:.2f}, {self.source})")
    
    def token_estimate(self) -> int:
        return int(len(self.content.split()) * 1.3)

@dataclass
class ContextOperation:
    operation_type: OperationType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    operation_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    result_queue: Optional[queue.Queue] = None

class LLMContextManager(dspy.Module):
    """Real LLM modules for reasoning with proper DSPy usage - Azure OpenAI compatible"""
    
    def __init__(self):
        super().__init__()
        
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


        def generate_initial_hypotheses(self, evidence_items: str) -> str:
            hypothesis_prompt = f"""
        {self.domain_config.hypothesis_instructions}
        
        Format each as: "HYPOTHESIS: [clear statement] | CONFIDENCE: [0.X] | REASONING: [why this makes sense]"
        
        EVIDENCE ITEMS:
        {evidence_items}
        
        CONTEXT: {self.domain_config.context_description}
        """
            
            hypotheses = self.generate_hypotheses(
                evidence_items=hypothesis_prompt,
                business_context=self.domain_config.context_description
            )
            return hypotheses.hypothesis_suggestions
        
    
    
####        def generate_initial_hypotheses(self, evidence_items: str) -> str:
####            """LLM generates initial hypotheses from evidence"""
####            
####            hypothesis_prompt = f"""
####        Generate 3-5 business hypotheses from evidence. Each hypothesis should:
####        1. Be testable against future evidence
####        2. Explain the observed evidence patterns
####        3. Include initial confidence (0.1 to 0.9)
####        4. Focus on business strategy, market conditions, operational efficiency
####        
####        Format each as: "HYPOTHESIS: [clear statement] | CONFIDENCE: [0.X] | REASONING: [why this makes sense]"
####        
####        EVIDENCE ITEMS:
####        {evidence_items}
####        
####        BUSINESS CONTEXT: Business decision support context with focus on strategy, operations, and market analysis
####        """
####            
####            hypotheses = self.generate_hypotheses(
####                evidence_items=hypothesis_prompt,
####                business_context="Business decision support context"
####            )
####            return hypotheses.hypothesis_suggestions
####

    def reason_about_context_changes(self, context_summary: str, new_items: str) -> str:
        """Direct Azure OpenAI call for guaranteed JSON mode with better prompting"""
        import requests
        import os
        
        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_VERSION')}"
        
        headers = {
            "api-key": os.getenv("AZURE_OPENAI_KEY"),
            "Content-Type": "application/json"
        }
        
        # IMPROVED PROMPT: Explicit instruction to update confidence based on evidence
        prompt = f"""You are analyzing business hypotheses based on new evidence. Your job is to UPDATE confidence scores based on how the new evidence supports or contradicts each hypothesis.
    
    CURRENT HYPOTHESES WITH CONFIDENCE SCORES:
    {context_summary}
    
    NEW EVIDENCE TO ANALYZE:
    {new_items}
    
    INSTRUCTIONS:
    1. For each hypothesis, determine if the new evidence SUPPORTS, CONTRADICTS, or is NEUTRAL
    2. INCREASE confidence (by 0.1-0.3) for hypotheses supported by evidence
    3. DECREASE confidence (by 0.1-0.3) for hypotheses contradicted by evidence  
    4. Keep confidence unchanged only if evidence is truly neutral
    5. Consider these specific evidence impacts:
       - "Database optimization improved query performance by 60%" → SUPPORTS operational efficiency hypotheses
       - "Critical infrastructure failure detected" → CONTRADICTS operational efficiency hypotheses
       - "Enterprise customers reporting satisfaction improvements" → SUPPORTS customer-focused hypotheses
       - "SMB customer acquisition costs increased 50%" → CONTRADICTS cost efficiency hypotheses
       - "New competitor entered SMB market" → SUPPORTS market volatility/competition hypotheses
    
    RESPOND WITH JSON in this EXACT format:
    {{
      "hypotheses": [
        {{
          "statement": "exact hypothesis text",
          "confidence": updated_confidence_score,
          "status": "active",
          "reasoning": "why confidence changed based on evidence"
        }}
      ],
      "new_hypotheses": []
    }}
    
    IMPORTANT: Confidence scores MUST be different from input if evidence is relevant. Do not just echo the same confidence values."""
    
        data = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

####    def reason_about_context_changes(self, context_summary: str, new_items: str) -> str:
####        """Direct Azure OpenAI call for guaranteed JSON mode"""
####        import requests
####        import os
####        
####        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_VERSION')}"
####        
####        headers = {
####            "api-key": os.getenv("AZURE_OPENAI_KEY"),
####            "Content-Type": "application/json"
####        }
####        
####        data = {
####            "messages": [
####                {"role": "user", "content": f"""Analyze context and respond with JSON:
####    
####    {context_summary}
####    
####    New items: {new_items}
####    
####    Respond with JSON: {{"hypotheses": [], "new_hypotheses": []}}"""}
####            ],
####            "response_format": {"type": "json_object"},
####            "temperature": 0.3,
####            "max_tokens": 2000
####        }
####        
####        response = requests.post(url, headers=headers, json=data)
####        return response.json()["choices"][0]["message"]["content"]        



    def generate_initial_hypotheses(self, evidence_items: str) -> str:
        """LLM generates initial hypotheses from evidence"""
        
        # Include instructions in the input data for Azure OpenAI compatibility
        hypothesis_prompt = f"""
Generate 3-5 business hypotheses from evidence. Each hypothesis should:
1. Be testable against future evidence
2. Explain the observed evidence patterns
3. Include initial confidence (0.1 to 0.9)
4. Focus on business strategy, market conditions, operational efficiency
5. Be contradictory/alternative to other hypotheses

Format each as: "HYPOTHESIS: [clear statement] | CONFIDENCE: [0.X] | REASONING: [why this makes sense]"

EVIDENCE ITEMS:
{evidence_items}

BUSINESS CONTEXT: Business decision support context with focus on strategy, operations, and market analysis
"""
        
        hypotheses = self.generate_hypotheses(
            evidence_items=hypothesis_prompt,
            business_context="Business decision support context"
        )
        return hypotheses.hypothesis_suggestions

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

class DomainConfig:
    def __init__(self, domain_type: str = "business"):
        if domain_type == "business":
            self.keywords = {
                'sales': ['sales', 'revenue', 'deals'],
                'technical': ['server', 'performance', 'system'],
                'market': ['market', 'competitor'],
                'enterprise': ['enterprise', 'b2b'],
                'smb': ['smb', 'small', 'medium']
            }
            self.context_description = "Business decision support context with focus on strategy, operations, and market analysis"
            self.hypothesis_instructions = "Generate 3-5 business hypotheses from evidence. Focus on business strategy, market conditions, operational efficiency"
            self.executive_role = "CEO/senior leadership team"
            self.output_format = "executive briefing"
            
        elif domain_type == "criminal_investigation":
            self.keywords = {
                'evidence': ['forensic', 'dna', 'fingerprint', 'witness'],
                'temporal': ['timeline', 'alibi', 'when', 'time'],
                'relationship': ['suspect', 'victim', 'family', 'friend'],
                'location': ['scene', 'location', 'address', 'where'],
                'motive': ['motive', 'reason', 'why', 'conflict']
            }
            self.context_description = "Criminal investigation context focusing on evidence analysis, suspect identification, and case resolution"
            self.hypothesis_instructions = "Generate 3-5 investigative hypotheses from evidence. Focus on motive, opportunity, means, and suspect identification"
            self.executive_role = "detective/investigation team"
            self.output_format = "case summary"

class AsyncReasoningEngine:
    """
    Asynchronous reasoning engine with separate context monitoring
    - Fast assertion processing (no blocking)
    - Separate reasoning cycles that monitor context changes
    - Reasoning happens between assertion batches
    - Proper DSPy usage with clean data separation
    - Simple loop counter failsafe to prevent infinite reasoning
    """
    

    def __init__(self, max_context_tokens: int = 4000, domain: str = "business"):

        # set investigation domain
        self.domain_config = DomainConfig(domain)
        
        # Setup LLM
        setup_dspy()
        
        # Core state
        self.context_items: List[ContextItem] = []
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = int(max_context_tokens * 0.8)
        self.running = False
        
        # Separate threads for different concerns
        self.queued_assertions = queue.Queue()  # Single queue for all operations
        self.context_thread = None              # Fast context updates
        self.reasoning_thread = None            # Slow reasoning cycles
        
        # Context change tracking
        self.context_version = 0          # Incremented on each context change
        self.last_reasoned_version = 0    # Last version that was reasoned about
        self.context_change_event = threading.Event()  # Signals reasoning needed
        
        # ADDED: Simple loop counter failsafe
        self.reasoning_loop_count = 0     # Simple counter for current reasoning session
        self.max_reasoning_loops = 10     # Hard stop after 10 loops in one session
        
        # Reasoning cycle control
        self.last_context_hash = None       # Hash of context after last reasoning cycle
        self.context_hash_history = []      # Track hash history for oscillation detection
        self.max_sterile_cycles = 3         # Stop after 3 cycles with no context changes
        self.consecutive_sterile_cycles = 0
        self.max_reasoning_cycles = 10       # Hard limit on reasoning cycles (legacy)
        self.oscillation_detection_window = 6  # Check last 6 cycles for patterns
        
        # Deep thought monitoring
        self.deep_thought_mode = False
        self.deep_thought_start = None
        self.max_deep_thought_minutes = 5    # Alert after 5 minutes of deep thought
        self.human_override_event = threading.Event()  # Allow human interruption
        self.show_full_context = False      # Option to show full context each cycle
        
        # Deep thought isolation
        self.assertion_batch_size = 5           # Process in batches
        
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
        
        # LLM components
        self.llm_manager = LLMContextManager()
        
    def _extract_key_phrases(self, content: str) -> set:
        """Extract key phrases from hypothesis content for better matching"""
        content_lower = content.lower()
        
        # Define important business phrases that indicate hypothesis topics
        key_phrases = set()
        
        business_phrases = [
            'marketing campaign', 'sales increase', 'seasonal demand', 'operational efficiency',
            'customer acquisition', 'market share', 'competitor', 'infrastructure',
            'performance improvement', 'customer satisfaction', 'q4 sales', 'enterprise',
            'smb', 'revenue', 'target', 'growth strategy', 'promotion', 'discount'
        ]
        
        for phrase in business_phrases:
            if phrase in content_lower:
                key_phrases.add(phrase)
        
        # Also extract significant word pairs
        words = content_lower.split()
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            # Include pairs that contain important business terms
            if any(term in pair for term in ['sales', 'marketing', 'customer', 'operational', 'performance', 'revenue']):
                key_phrases.add(pair)
        
        return key_phrases

    def _hypothesis_matches(self, existing_content: str, llm_content: str) -> bool:
        """Check if LLM hypothesis content matches an existing hypothesis with multiple strategies"""
        
        if not existing_content or not llm_content:
            return False
        
        existing_lower = existing_content.lower()
        llm_lower = llm_content.lower()
        
        #print(f"   DEBUG MATCH: Comparing:")
        #print(f"   DEBUG MATCH:   Existing: {existing_lower[:60]}...")
        #print(f"   DEBUG MATCH:   LLM:      {llm_lower[:60]}...")
        
        # 1. Exact substring match (either direction)
        if llm_lower in existing_lower or existing_lower in llm_lower:
            #print(f"   DEBUG MATCH: SUBSTRING MATCH FOUND")
            return True
        
        # 2. Key phrase matching
        existing_key_phrases = self._extract_key_phrases(existing_content)
        llm_key_phrases = self._extract_key_phrases(llm_content)
        
        phrase_overlap = existing_key_phrases.intersection(llm_key_phrases)
        #print(f"   DEBUG MATCH: Key phrase overlap: {phrase_overlap}")
        
        if len(phrase_overlap) >= 2:  # At least 2 key phrases match
            #print(f"   DEBUG MATCH: KEY PHRASE MATCH FOUND")
            return True
        
        # 3. Relaxed word overlap (lower threshold)
        existing_words = set(existing_lower.split())
        llm_words = set(llm_lower.split())
        
        # Remove common stop words that don't help with matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'that', 'this'}
        existing_words = existing_words - stop_words
        llm_words = llm_words - stop_words
        
        if len(existing_words) == 0 or len(llm_words) == 0:
            #print(f"   DEBUG MATCH: No meaningful words after stop word removal")
            return False
        
        intersection = existing_words.intersection(llm_words)
        similarity = len(intersection) / max(len(existing_words), len(llm_words))
        
        #print(f"   DEBUG MATCH: Word overlap: {intersection}")
        #print(f"   DEBUG MATCH: Similarity score: {similarity:.2f} (threshold: 0.4)")
        
        if similarity > 0.4:  # Lowered from 0.6 to 0.4
            print(f"   DEBUG MATCH: WORD OVERLAP MATCH FOUND")
            return True
        
        #print(f"   DEBUG MATCH: NO MATCH FOUND")
        return False
    
####    def _hypothesis_matches(self, existing_content: str, llm_content: str) -> bool:
####        """Check if LLM hypothesis content matches an existing hypothesis"""
####        # Simple similarity check - could be enhanced
####        existing_words = set(existing_content.lower().split())
####        llm_words = set(llm_content.lower().split())
####        
####        # If most words match, consider it the same hypothesis
####        if len(existing_words) == 0 or len(llm_words) == 0:
####            return False
####        
####        intersection = existing_words.intersection(llm_words)
####        similarity = len(intersection) / max(len(existing_words), len(llm_words))
####        
####        return similarity > 0.6  # 60% word overlap threshold
####
####        print("Async reasoning engine initialized - ready to start")
    
    def _track_reasoning_changes(self, before_items: List[ContextItem]) -> Dict[str, Any]:
        """Track what changed during reasoning cycle"""
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
    
    def start(self):
        """Start both context processing and reasoning monitoring"""
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
        print("   Context processor: handles assertions quickly")
        print("   Reasoning monitor: watches for context changes")
    
    def stop(self):
        """Stop all processing"""
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
        """Add evidence to the single queue"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        operation = ContextOperation(
            operation_type=OperationType.ADD_EVIDENCE,
            data={'content': content, 'source': source, 'confidence': confidence}
        )
        
        self.queued_assertions.put(operation)
        print(f"Evidence queued: {content[:50]}...")
        return operation.operation_id
    
    def add_hypothesis(self, content: str, confidence: float = 0.6) -> str:
        """Add hypothesis to the single queue"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        operation = ContextOperation(
            operation_type=OperationType.ADD_HYPOTHESIS,
            data={'content': content, 'confidence': confidence}
        )
        
        self.queued_assertions.put(operation)
        print(f"Hypothesis queued: {content[:50]}...")
        return operation.operation_id
    
    def clear_context(self, base_name: str = "investigation") -> str:
        """Clear all context - generates investigation ID and queues to single queue"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        # Generate unique investigation ID
        from datetime import datetime
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
    
    def get_investigation_results(self, investigation_id: str) -> Dict[str, Any]:
        """Get investigation results by ID (SQLite stub)"""
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
    
    def get_investigation_results(self, investigation_id: str) -> Dict[str, Any]:
        """Get investigation results by ID (SQLite stub)"""
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

    def query_context_sync(self, query: str, timeout: float = 30.0) -> Optional[str]:
        """Query reasoning state synchronously"""
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
    
    def toggle_full_context_display(self, show: bool = True):
        """Toggle display of full context window at each reasoning cycle"""
        self.show_full_context = show
        print(f"Full context display: {'ENABLED' if show else 'DISABLED'}")
    
    def force_stop_reasoning(self, reason: str = "Human override"):
        """Allow external process to force reasoning to stop"""
        print(f"FORCED REASONING STOP: {reason}")
        print(f"   Was on loop {self.reasoning_loop_count} of {self.max_reasoning_loops}")
        self.human_override_event.set()
        self.deep_thought_mode = False

    def configure_max_loops(self, max_loops: int):
        """Set maximum reasoning loops per session"""
        self.max_reasoning_loops = max_loops
        print(f"Maximum reasoning loops set to: {max_loops}")

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Get current status"""
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
                'reasoning_loops_in_session': len([h for h in self.context_hash_history if h != self.context_hash_history[0]]) if self.context_hash_history else 0,
                
                # Simple loop counter
                'reasoning_loop_count': self.reasoning_loop_count,
                'max_reasoning_loops': self.max_reasoning_loops,
                
                'stats': self.stats.copy()
            }
    
    def _context_processing_loop(self):
        """Process operations from single queue, with clear boundary logic"""
        print("Context processing started (single queue)")
        
        while self.running:
            try:
                # Check if clear is at head of queue (after deep thought completes)
                if not self.deep_thought_mode and not self.queued_assertions.empty():
                    # Peek at next operation without removing it
                    temp_ops = []
                    try:
                        next_op = self.queued_assertions.get_nowait()
                        temp_ops.append(next_op)
                        
                        if next_op.operation_type == OperationType.CLEAR_CONTEXT:
                            # Process clear immediately (capture results, clear context, consume operation)
                            print("Processing clear at head of queue")
                            self._process_context_operation(next_op)
                            continue
                        else:
                            # Put it back, we'll process in batch
                            self.queued_assertions.put(next_op)
                    except queue.Empty:
                        pass
                
                # Process operations up to first clear (if any)
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
                
                # Process the operations before clear boundary
                if operations_to_process:
                    for operation in operations_to_process:
                        if operation.operation_type == OperationType.SHUTDOWN:
                            return
                        
                        print(f"Processing: {operation.operation_type.value}")
                        self._process_context_operation(operation)
                        
                        # Signal context change for evidence/hypothesis operations
                        if operation.operation_type in [OperationType.ADD_EVIDENCE, OperationType.ADD_HYPOTHESIS]:
                            self._signal_context_change(is_external=True)
                
                # If no operations to process, wait briefly
                if not operations_to_process:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Context processing error: {e}")
                time.sleep(1)
        
        print("Context processing stopped")
    
    def _reasoning_monitoring_loop(self):
        """Separate reasoning process - monitors context changes"""
        print("Reasoning monitor started (watches for context changes)")
        
        while self.running:
            try:
                # Wait for context changes or timeout
                self.context_change_event.wait(timeout=5.0)
                
                if not self.running:
                    break
                
                # Check if reasoning is needed
                if self.context_version > self.last_reasoned_version:
                    # Enter deep thought mode if multiple cycles expected
                    if not self.deep_thought_mode:
                        self.deep_thought_mode = True
                        self.deep_thought_start = datetime.now()
                        self.reasoning_loop_count = 0  # RESET simple counter
                        print("ENTERING DEEP THOUGHT MODE...")
                    
                    reasoning_result = self._perform_reasoning_cycle()
                    
                    # Check various stop conditions
                    if reasoning_result['should_stop_reasoning']:
                        print(f"Reasoning stopped: {reasoning_result['stop_reason']}")
                        self.deep_thought_mode = False
                        
                    # Check for human override
                    elif self.human_override_event.is_set():
                        print("Reasoning stopped: Human override requested")
                        self.human_override_event.clear()
                        self.deep_thought_mode = False
                        
                    # Check deep thought timeout
                    elif self.deep_thought_mode and self.deep_thought_start:
                        deep_thought_duration = (datetime.now() - self.deep_thought_start).total_seconds()
                        if deep_thought_duration > (self.max_deep_thought_minutes * 60):
                            print(f"DEEP THOUGHT TIMEOUT after {deep_thought_duration/60:.1f} minutes")
                            print("Pausing reasoning - waiting for external intervention or new assertions")
                            self.deep_thought_mode = False
                            # Don't exit loop, just wait for new assertions
                else:
                    # No reasoning needed, exit deep thought
                    if self.deep_thought_mode:
                        print("EXITING DEEP THOUGHT MODE - reasoning complete")
                        self._generate_deep_thought_summary()
                        self.deep_thought_mode = False
                        
                        # Process any assertions that were queued during deep thought
                        # This is now handled by the main context processing loop
                        print("Deep thought complete - context processing will resume")
                
                # Clear the event
                self.context_change_event.clear()
                
            except Exception as e:
                print(f"Reasoning monitor error: {e}")
                time.sleep(1)
        
        print("Reasoning monitor stopped")
    
    def _process_context_operation(self, operation: ContextOperation):
        """Process context operations quickly (no LLM calls)"""
        
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
        """Add evidence to context quickly"""
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
        """Add hypothesis to context quickly"""
        hypothesis_item = ContextItem(
            content=data['content'],
            timestamp=datetime.now(),
            item_type=ItemType.HYPOTHESIS,
            status=Status.ACTIVE,
            confidence=data['confidence'],
            importance=0.9,
            source="external_hypothesis",
            tags=["hypothesis"],
            reasoning_version=0
        )
        
        with self.context_lock:
            self.context_items.append(hypothesis_item)
        
        print(f"Hypothesis added: {data['content'][:50]}...")
    
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
        """Store investigation results in SQLite (stub)"""
        # TODO: Store in SQLite database with TTL
        print(f"STUB: Storing results for investigation {investigation_id}")
        print(f"   Hypotheses: {results['hypotheses_count']}")
        print(f"   Evidence: {results['evidence_count']}")
        print(f"   Reasoning cycles: {results['reasoning_cycles']}")
        if results['hypotheses']:
            top_hypothesis = max(results['hypotheses'], key=lambda x: x['confidence'])
            print(f"   Top hypothesis: {top_hypothesis['content'][:60]}... (confidence: {top_hypothesis['confidence']:.2f})")
    
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
    {chr(10).join([f"• {e.content}" for e in evidence[:5]])}
    
    EXECUTIVE QUERY: {query}
    
    INSTRUCTIONS:
    Provide a crisp, executive-ready response with:
    1. BOTTOM LINE UP FRONT: One clear conclusion/recommendation
    2. CONFIDENCE LEVEL: Specific percentage or "high/medium/low" with reasoning
    3. KEY SUPPORTING FACTS: 2-3 bullet points of strongest evidence
    4. BUSINESS IMPLICATIONS: What this means for strategy/operations
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
        """Signal that context has changed (triggers reasoning)"""
        with self.context_lock:
            self.context_version += 1
            # Reset sterile cycles when new external assertions arrive
            if is_external:
                self.consecutive_sterile_cycles = 0
            self.stats['context_changes'] += 1
        
        # Wake up reasoning thread
        self.context_change_event.set()
    

    
    def _perform_reasoning_cycle(self) -> Dict[str, Any]:
        """Perform reasoning cycle with simple loop counter failsafe"""
        
        # INCREMENT SIMPLE COUNTER
        self.reasoning_loop_count += 1
        
        print(f"\nREASONING CYCLE {self.reasoning_loop_count} STARTED")
        print(f"   Loop {self.reasoning_loop_count} of {self.max_reasoning_loops} max")
        
        # display context window, formatted
        self._display_full_context_window()

        # display context window, raw
        with self.context_lock:
            raw_context = self._build_context_summary()
            print(f"\nRAW CONTEXT SENT TO LLM:")
            print(f"{'='*60}")
            print(raw_context)
            print(f"{'='*60}")

        # SIMPLE FAILSAFE CHECK FIRST
        if self.reasoning_loop_count >= self.max_reasoning_loops:
            return {
                'should_stop_reasoning': True,
                'stop_reason': f'FAILSAFE: Maximum reasoning loops reached ({self.max_reasoning_loops})',
                'context_changed': False,
                'failsafe_triggered': True
            }
        
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
        
        with self.context_lock:
            cycle_start_version = self.context_version
            new_items = [item for item in self.context_items 
                          if item.reasoning_version < cycle_start_version]


            
            if not new_items:
                return {
                    'should_stop_reasoning': True,
                    'stop_reason': 'No new items to process',
                    'context_changed': False
                }
            
            # Additional check: Don't reason if context is empty
            if len(self.context_items) == 0:
                return {
                    'should_stop_reasoning': True,
                    'stop_reason': 'Context is empty - nothing to reason about',
                    'context_changed': False
                }
            
            context_summary = self._build_context_summary()
            hypotheses = [i for i in self.context_items if i.item_type == ItemType.HYPOTHESIS]
        
        print(f"   Processing {len(new_items)} new items")
        print(f"   Current hypotheses: {len(hypotheses)}")
        print(f"   Consecutive sterile cycles: {self.consecutive_sterile_cycles}")
        
        try:
            # Perform reasoning operations
            if not hypotheses and new_items:
                print(f"   Generating initial hypotheses from evidence...")
                self._generate_initial_hypotheses(new_items)
            elif hypotheses and new_items:
                print(f"   Analyzing new evidence against existing hypotheses...")
                self._reason_about_new_items(context_summary, new_items)
            
            # Compression if needed
            total_tokens = sum(item.token_estimate() for item in self.context_items)
            if total_tokens > self.compression_threshold:
                print(f"   Performing context compression...")
                self._perform_compression()
            
            # Mark items as processed
            with self.context_lock:
                for item in new_items:
                    item.reasoning_version = cycle_start_version
                self.last_reasoned_version = cycle_start_version
            
            # Track and report changes
            changes = self._track_reasoning_changes(before_items)
            self._report_reasoning_progress(self.reasoning_loop_count, changes)
            
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
            
            # Generate situation report at end of cycle
            situation_report = self._generate_situation_report(new_items)
            print(situation_report)
            
            # Show full context if enabled
            if self.show_full_context:
                self._display_full_context_window()
            
            return {
                'should_stop_reasoning': False,
                'stop_reason': None,
                'context_changed': context_changed
            }
            
        except Exception as e:
            print(f"Reasoning cycle failed: {e}")
            return {
                'should_stop_reasoning': True,
                'stop_reason': f'Reasoning error: {e}',
                'context_changed': False
            }
    
    def _generate_deep_thought_summary(self):
        """Generate summary of what was accomplished during deep thought session"""
        
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
                print(f"   Primary Conclusion: {top_theory.content}")  # Show full content
            
            print(f"   Confidence Level: {top_theory.confidence:.2f} ({self._confidence_description(top_theory.confidence)})")
            
            if len(active_hyp) > 1:
                other_theories = [h for h in active_hyp if h != top_theory]
                print(f"\n   Secondary Theories ({len(other_theories)}):")
                for i, theory in enumerate(other_theories, 1):
                    if "LLM-generated hypothesis based on evidence patterns" in theory.content:
                        print(f"   {i}. PLACEHOLDER: {theory.content}")
                    else:
                        print(f"   {i}. {theory.content}")  # Show full content
                    print(f"      Confidence: {theory.confidence:.2f}")
        
        # What changed during session
        if revivals:
            print(f"\nSESSION INSIGHTS:")
            print(f"   Theories Revived ({len(revivals)}):")
            for i, revival in enumerate(revivals, 1):
                print(f"   {i}. {revival.content}")  # Show full revival content
        
        if dormant_hyp:
            print(f"\n   Theories Set Aside ({len(dormant_hyp)}):")
            for i, dormant in enumerate(dormant_hyp, 1):
                print(f"   {i}. {dormant.content}")  # Show full dormant content
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
    
    def _compute_context_hash(self) -> str:
        """Compute hash of entire context state"""
        import hashlib
        
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
    
    def _generate_situation_report(self, new_items: List[ContextItem]) -> str:
        """Generate 'where are we' summary like a police detective reporting to captain"""
        
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
                report += f"   {i}. {confidence_trend} {hyp.content}\n"  # Show full content
                report += f"      Confidence: {hyp.confidence:.2f}\n"
        
        # What just happened (new evidence impact)
        if new_items:
            evidence_items = [item for item in new_items if item.item_type == ItemType.EVIDENCE]
            if evidence_items:
                report += f"\nNEW DEVELOPMENTS ({len(evidence_items)} items):\n"
                for i, evidence in enumerate(evidence_items, 1):
                    age_minutes = (datetime.now() - evidence.timestamp).total_seconds() / 60
                    report += f"   {i}. {evidence.content}\n"  # Show full content
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
                    report += f"      {i}. {hyp.content}\n"  # Show full content
                    report += f"         Confidence: {hyp.confidence:.2f}\n"
            
            if dormant_hyp:
                report += f"   DORMANT THEORIES: {len(dormant_hyp)} theories shelved (still recoverable)\n"
        
        # Current focus and next steps
        report += f"\nCURRENT FOCUS:\n"
        if active_hyp:
            top_theory = max(active_hyp, key=lambda x: x.confidence)
            report += f"   Thesis with highest confidence:\n"
            report += f"     {top_theory.content}\n"  # Show full content
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
    
    def _generate_initial_hypotheses(self, evidence_items: List[ContextItem]):
        """Generate initial hypotheses from evidence using Azure OpenAI compatible DSPy"""
        print("   Generating initial hypotheses...")
        
        # Build clean evidence data for DSPy
        evidence_summary = "\n".join([item.to_tuple_string() for item in evidence_items])
        
        # Azure OpenAI compatible DSPy usage
        hypothesis_suggestions = self.llm_manager.generate_initial_hypotheses(
            evidence_items=evidence_summary
        )
        
        print(f"   LLM Hypothesis Suggestions: {hypothesis_suggestions[:100]}...")
        
        # Parse and create hypotheses (programmatic part)
        self._create_hypotheses_from_suggestions(hypothesis_suggestions)
        
        self.stats['hypotheses_generated'] += 1
    
    def _reason_about_new_items(self, context_summary: str, new_items: List[ContextItem]) -> bool:
        """Reason about new items using Azure OpenAI compatible DSPy - returns True if changes made"""
        print("   Reasoning about new items against existing context...")
        
        # Build clean new items data for DSPy
        new_items_summary = "\n".join([item.to_tuple_string() for item in new_items])
        
        # Azure OpenAI compatible - instructions included in data
        reasoning_analysis = self.llm_manager.reason_about_context_changes(
            context_summary=context_summary,
            new_items=new_items_summary
        )
        
        print(f"   LLM Reasoning: {reasoning_analysis[:100]}...")
        
        # Apply reasoning results and track if changes were made (programmatic part)
        changes_made = self._apply_reasoning_analysis(reasoning_analysis, new_items)
        
        return changes_made
    
    def _apply_reasoning_analysis(self, analysis: str, new_items: List[ContextItem]) -> bool:
        """Apply LLM reasoning decisions directly to context"""
        import json
        
        #print(f"   DEBUG: Full LLM response length: {len(analysis)}")
        
        try:
            decisions = json.loads(analysis)
            #print(f"   DEBUG: Parsed JSON keys: {decisions.keys()}")
            #print(f"   DEBUG: Hypotheses in response: {len(decisions.get('hypotheses', []))}")
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
                    
                    # Match hypothesis by content similarity
                    if self._hypothesis_matches(item.content, llm_content):
                        #print(f"   DEBUG: MATCHED '{llm_content[:50]}...' -> '{item.content[:50]}...'")
                        
                        old_confidence = item.confidence
                        old_status = item.status
                        
                        # Apply LLM's decisions
                        item.confidence = float(llm_confidence)
                        
                        if llm_status == 'active':
                            item.status = Status.ACTIVE
                        elif llm_status == 'weakened':
                            item.status = Status.WEAKENED
                        elif llm_status == 'dormant':
                            item.status = Status.DORMANT
                        
                        #print(f"   DEBUG: Updated confidence: {old_confidence:.2f} -> {item.confidence:.2f}")
                        
                        # Track changes
                        confidence_change = abs(old_confidence - item.confidence)
                        if confidence_change > 0.05:
                            changes_made = True
                        
                        if old_status != item.status:
                            changes_made = True
                            if old_status == Status.DORMANT and item.status == Status.ACTIVE:
                                self._log_revival(item, "LLM revival: Supporting evidence detected")
                        
                        matched_hypotheses.add(id(item))  # Mark as matched
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
        
        #print(f"   DEBUG: Changes made during reasoning: {changes_made}")
        return changes_made

    def _create_hypotheses_from_suggestions(self, suggestions: str):
        """Create hypothesis items from LLM suggestions with improved parsing"""
        print(f"   Processing LLM Hypothesis Suggestions:")
        print(f"   Raw LLM output: {suggestions}")
        
        # Parse the LLM suggestions for actual hypothesis statements
        hypotheses_created = []
        
        # IMPROVED: Look for the structured format from DSPy instructions
        suggestion_lines = suggestions.split('\n')
        
        for line in suggestion_lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or len(line) < 10:
                continue
            
            # Look for structured format: "HYPOTHESIS: ... | CONFIDENCE: ... | REASONING: ..."
            if 'HYPOTHESIS:' in line.upper() and '|' in line:
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
                        hypotheses_created.append(hypothesis_item)
                    
                    continue
            
            # Fallback: Look for hypothesis patterns in unstructured response
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
            
            # Also capture sentences that look like business hypotheses
            business_keywords = ['enterprise', 'smb', 'customer', 'revenue', 'performance', 'market', 'sales']
            if (not is_hypothesis and 
                len(line) > 20 and 
                any(keyword in line.lower() for keyword in business_keywords) and
                ('should' in line.lower() or 'will' in line.lower() or 'is' in line.lower())):
                is_hypothesis = True
                hypothesis_content = line
            
            if is_hypothesis and len(hypothesis_content) > 15:
                # Clean up common prefixes
                prefixes_to_remove = ['- ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ']
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
                    hypotheses_created.append(hypothesis_item)
        
        # Report what was created
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
                    content=f"Business pattern analysis suggests: {suggestions[:100]}...",
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
    
    def _log_revival(self, hypothesis: ContextItem, reason: str):
        """Log hypothesis revival"""
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
    
    def _perform_compression(self):
        """Perform context compression"""
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
        """Build context summary for LLM"""
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
        """Quick importance assessment"""
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
        
def demo_async_reasoning():
    """Demonstrate async reasoning engine"""
    
    print("ASYNCHRONOUS CONTEXT-WATCHING REASONING ENGINE")
    print("=" * 70)
    
    try:
        # Initialize and start

        # Business reasoning (default)
        engine = AsyncReasoningEngine(domain="business", max_context_tokens=3500)

        # Criminal investigation
        #engine = AsyncReasoningEngine(domain="criminal_investigation", max_context_tokens=3500)

        #engine = AsyncReasoningEngine(max_context_tokens=3500)
        engine.start()
        
        # Show initial status
        status = engine.get_status_snapshot()
        print(f"Initial Status: Engine running, waiting for assertions...")
        
        print("\nTESTING ASYNC ASSERTION PROCESSING:")
        
        # Start with clear to establish investigation
        initial_investigation_id = engine.clear_context("demo_investigation")
        print(f"Started investigation: {initial_investigation_id}")
        
        # Add some initial evidence quickly
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
            #print(f"   Added in {duration*1000:.1f}ms: {content[:50]}...")
            time.sleep(0.5)  # Brief pause between additions
        
        print("\n   Initial evidence added - monitoring loop will trigger reasoning when ready...")

        print("\n2. Testing queries:")

        # Show status after reasoning
        status = engine.get_status_snapshot()
        #print(f"\nStatus after initial reasoning:")
        #print(f"   Total items: {status['total_items']}")
        #print(f"   Hypotheses: {status['hypotheses']}")
        #print(f"   Reasoning cycles: {status['stats']['reasoning_cycles']}")
        #print(f"   Deep thought mode: {status['deep_thought_mode']}")
        #print(f"   Loop counter: {status['reasoning_loop_count']}/{status['max_reasoning_loops']}")
        
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
            #print(f"Added: {content}")
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

        ## Test context clearing with investigation IDs
        #print("\n5. Testing context clearing with investigation tracking:")
        #print(f"   Before clear: {len(engine.context_items)} items")
        #print(f"   Deep thought mode: {engine.deep_thought_mode}")
        
        #investigation_id = engine.clear_context("demo_investigation")
        #print(f"   Investigation ID: {investigation_id}")
        
        ## Wait for context clear to be processed (after deep thought completes)
        #print("   Waiting for context clear to be processed...")
        #start_wait = time.time()
        #while len(engine.context_items) > 0 and (time.time() - start_wait) < 10:
        #    time.sleep(0.5)
        #    print(".", end="", flush=True)
        #print()
        
        #print(f"   After clear: {len(engine.context_items)} items")
        #print(f"   Deep thought mode: {engine.deep_thought_mode}")
        
        # Test getting investigation results
        #print(f"\n6. Testing investigation results retrieval:")
        #results = engine.get_investigation_results(investigation_id)
        #print(f"   Retrieved results for: {results.get('investigation_id')}")
        
        # Also test retrieving results from the initial investigation
        #print(f"\n7. Testing initial investigation results:")
        #initial_results = engine.get_investigation_results(initial_investigation_id)
        #print(f"   Retrieved initial results for: {initial_results.get('investigation_id')}")

        print("\nDEMONSTRATION COMPLETE")

        """
        print("Entering production monitoring mode...")
        print("Engine ready for new assertions. Press Ctrl+C to exit.")

        # Production monitoring loop with better exception handling
        try:
            cycle_count = 0
            while engine.running:
                time.sleep(5)
                cycle_count += 1
                
                # Periodic status log (every 30 seconds)
                if cycle_count % 6 == 0:
                    status = engine.get_status_snapshot()
                    print(f"Status: {status['total_items']} items, "
                        f"{status['hypotheses']} hypotheses, "
                        f"{'deep thought' if status['deep_thought_mode'] else 'waiting'}")

        except KeyboardInterrupt:
            print("\nShutdown signal received")
        except Exception as e:
            print(f"Unexpected error: {e}")

        """

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
                
                # Final status
                final_status = engine.get_status_snapshot()
                #print(f"\nFinal Status:")
                #print(f"   Total items: {final_status['total_items']}")
                #print(f"   Hypotheses: {final_status['hypotheses']}")
                #print(f"   Total reasoning cycles: {final_status['stats']['reasoning_cycles']}")
                #print(f"   Assertions processed: {final_status['stats']['assertions_processed']}")
                #print(f"   Loop counter used: {final_status['reasoning_loop_count']}/{final_status['max_reasoning_loops']}")

                
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

        #engine = AsyncReasoningEngine(max_context_tokens=5000)
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
    
if __name__ == "__main__":
    print("Run with 'python -u reasoning.py' for real-time output")
    demo_async_reasoning()


