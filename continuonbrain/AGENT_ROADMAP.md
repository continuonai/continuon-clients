# Agent Manager Roadmap

## Current Status

The Agent Manager has been successfully transformed into a unified architecture where HOPE brain serves as the primary agent with hierarchical LLM fallback and active learning capabilities.

**Active Features:**
- ✅ HOPE-first hierarchical response system
- ✅ Dynamic model selection UI (Mock, Gemma 2B, 4B)
- ✅ Hot-swap model switching without restart
- ✅ Automatic conversation storage and learning
- ✅ Memory-enhanced LLM responses

**Current Metrics:**
- 67% of queries answered by HOPE directly
- 12 conversations learned and stored
- 3 models detected and available

---

## Robustness Improvements

### Critical Priority (Do First)

#### 1. Semantic Search for Memory Retrieval
**Status:** Planned  
**Current:** Keyword-based Jaccard similarity  
**Issue:** Misses semantically similar questions  
**Solution:** Implement sentence-transformers for semantic matching

```python
# services/experience_logger.py
from sentence_transformers import SentenceTransformer

class ExperienceLogger:
    def __init__(self, storage_dir):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB
        
    def _calculate_relevance(self, query, question):
        query_emb = self.encoder.encode(query)
        question_emb = self.encoder.encode(question)
        return cosine_similarity(query_emb, question_emb)
```

**Dependencies:** `pip install sentence-transformers`  
**Estimated Size:** 80MB model download

---

#### 2. Persistent Storage Migration
**Status:** Planned  
**Current:** `/tmp/continuonbrain_demo` (deleted on reboot)  
**Issue:** All learned knowledge lost on restart  
**Solution:** Move to permanent config directory

```python
# services/brain_service.py
self.experience_logger = ExperienceLogger(
    Path(config_dir) / "experiences"  # ~/.config/continuonbrain/experiences
)
```

**Files to modify:**
- `services/brain_service.py` - Update ExperienceLogger path
- Document storage location in README

---

#### 3. Conversation Deduplication
**Status:** Planned  
**Current:** Stores every response, including duplicates  
**Issue:** 8 of 12 current entries are duplicates  
**Solution:** Check similarity before storing

```python
def log_conversation(self, question, answer, ...):
    similar = self.get_similar_conversations(question, max_results=1)
    if similar and similar[0]['relevance'] > 0.95:
        logger.info(f"Skipping duplicate")
        return
    # Store new conversation
```

**Files to modify:**
- `services/experience_logger.py` - Add deduplication logic

---

### High Priority (Next Week)

#### 4. User Validation Feedback UI
**Status:** Planned  
**Current:** `"validated": false` never updated  
**Issue:** Can't distinguish good vs bad responses  
**Solution:** Add validation endpoint and UI

**New endpoint:**
```python
POST /api/agent/validate_response
{
  "timestamp": "2025-12-07T22:38:34.460706",
  "validated": true,
  "correction": "Actually, the correct answer is..."
}
```

**UI addition:** Add thumbs up/down in chat interface

**Files to create/modify:**
- `api/server.py` - Add validation endpoint
- `api/routes/ui_routes.py` - Add validation buttons to chat
- `services/experience_logger.py` - Update validation status

---

#### 5. HOPE Novelty-Based Confidence
**Status:** Planned  
**Current:** Hardcoded heuristics for capability queries  
**Issue:** Doesn't adapt to robot's actual knowledge  
**Solution:** Use HOPE's prediction error for confidence

```python
# services/agent_hope.py
def can_answer(self, message):
    encoded = self.brain.encode_input(message)
    with torch.no_grad():
        prediction = self.brain.forward(encoded)
        novelty = compute_prediction_error(prediction, encoded)
    confidence = 1.0 - novelty
    return (confidence >= self.threshold, confidence)
```

**Files to modify:**
- `services/agent_hope.py` - Replace heuristics with HOPE novelty
- `hope_impl/brain.py` - Add novelty computation method

---

#### 6. Learning Statistics Dashboard
**Status:** Planned  
**Purpose:** Monitor agent performance and knowledge growth

**New endpoint:**
```python
GET /api/agent/learning_stats
{
  "total_conversations": 12,
  "hope_response_rate": 0.67,
  "memory_hit_rate": 0.50,
  "by_agent": {
    "hope_brain": 8,
    "llm_with_hope_context": 3,
    "llm_only": 1
  },
  "knowledge_growth": [
    {"date": "2025-12-07", "count": 12}
  ]
}
```

**UI addition:** New dashboard page showing:
- HOPE response rate over time
- Topics learned (word cloud)
- Memory retrieval effectiveness

**Files to create/modify:**
- `api/server.py` - Add stats endpoint
- `api/routes/ui_routes.py` - Create learning dashboard page
- `services/experience_logger.py` - Add statistics methods

---

### Medium Priority (This Month)

#### 7. Memory Consolidation & Decay
**Status:** Planned  
**Purpose:** Manage memory quality over time

**Features:**
- Merge similar Q&A pairs
- Decay confidence of old/unused memories
- Remove low-quality entries
- Promote frequently accessed memories

```python
# services/experience_logger.py
def consolidate_memories(self, min_confidence=0.3):
    # 1. Group similar conversations
    # 2. Merge into single high-confidence entry
    # 3. Remove entries below threshold
    # 4. Update access counts
```

**Schedule:** Run daily or weekly via cron

---

#### 8. Conversation Context Window
**Status:** Planned  
**Current:** Each query independent  
**Issue:** Can't reference previous messages  
**Solution:** Maintain conversation sessions

```python
# services/brain_service.py
self.conversation_sessions = {}  # session_id -> history

def ChatWithGemma(self, message, history, session_id=None):
    if session_id:
        conversation_history = self.conversation_sessions[session_id]
        conversation_history.append({"role": "user", "content": message})
```

**Files to modify:**
- `services/brain_service.py` - Add session management
- `api/server.py` - Add session_id to chat endpoint
- `api/routes/ui_routes.py` - Send session_id from UI

---

### Low Priority (Future)

#### 9. Multi-Modal Learning
- Store image captures with conversations
- Link Q&A to robot's visual experiences
- "What did you see when I asked about X?"

#### 10. Topic Clustering
- Group conversations by topic
- Identify knowledge gaps
- Active learning suggestions

#### 11. Export Training Data
- Convert conversations to fine-tuning format
- Use for custom VLA training
- Enable when AI HAT+ available

---

## Performance Optimizations

### 1. Lazy Model Loading
Only initialize ExperienceLogger encoder when first needed

### 2. Memory Index
Build inverted index for faster lookup:
```python
{
  "camera": [conv_id_1, conv_id_3],
  "move": [conv_id_2, conv_id_5]
}
```

### 3. Batch Embedding
Encode multiple queries simultaneously when possible

---

## Testing Requirements

### Unit Tests Needed
- `tests/test_experience_logger.py` - Storage, retrieval, deduplication
- `tests/test_agent_hope.py` - Confidence assessment, memory retrieval
- `tests/test_semantic_search.py` - Embedding similarity

### Integration Tests Needed
- Full learning cycle (store → retrieve → enhance)
- Model switching with active conversations
- Session continuity across restarts

---

## Dependencies to Add

```bash
# For semantic search
pip install sentence-transformers

# For testing
pip install pytest pytest-asyncio

# For monitoring (optional)
pip install prometheus-client
```

---

## Migration Path

### Phase 1: Robustness (This Week)
1. Implement semantic search
2. Add deduplication
3. Move to persistent storage
4. Add validation UI

### Phase 2: Intelligence (This Month)
5. HOPE novelty-based confidence
6. Memory consolidation
7. Learning dashboard
8. Conversation context

### Phase 3: Production (When AI HAT+ Available)
9. Enable real Gemma models
10. Multi-modal learning
11. Custom VLA training
12. Full HOPE CMS integration

---

## Monitoring Checklist

- [ ] HOPE response rate trending up over time
- [ ] Memory hit rate > 30%
- [ ] Conversation storage growing steadily
- [ ] No duplicate entries being stored
- [ ] Validated responses > 50%
- [ ] Average confidence scores stable
- [ ] No memory/performance degradation

---

## Success Metrics

**Target State (3 Months):**
- HOPE answers 80%+ of capability queries
- Memory hit rate 50%+
- 500+ unique conversations learned
- 90%+ user validation rate
- <100ms memory retrieval latency

**Current State:**
- HOPE answers 67% of capability queries
- Memory hit rate 50%
- 12 conversations learned (8 unique)
- 0% validated (UI pending)
- ~50ms memory retrieval latency

---

Last Updated: 2025-12-07
