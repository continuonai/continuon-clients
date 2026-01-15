"""
Memory Consolidation System.

Implements CMS-style consolidation:
- Episodic → Semantic: Extract patterns from recent sessions
- Semantic → Procedural: Promote stable knowledge (rare, requires approval)

Analogous to sleep-time memory consolidation in biological systems.
"""

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from memory.manager import FilesystemMemory, MemoryLevel


@dataclass
class Pattern:
    """A pattern extracted from episodic memory."""
    key: str
    description: str
    frequency: int
    confidence: float
    source_episodes: list[str]
    first_seen: float
    last_seen: float


@dataclass
class ConsolidationResult:
    """Result of consolidation operation."""
    patterns_found: int
    patterns_promoted: int
    knowledge_updated: int
    duration_s: float


# Thresholds for consolidation
SEMANTIC_FREQUENCY_THRESHOLD = 3      # Min occurrences to promote to semantic
SEMANTIC_CONFIDENCE_THRESHOLD = 0.7   # Min confidence for semantic promotion
PROCEDURAL_AGE_THRESHOLD_DAYS = 30    # Min age for procedural consideration
PROCEDURAL_CONSISTENCY_THRESHOLD = 0.9  # Min consistency for procedural


class MemoryConsolidator:
    """
    Consolidates memories across the hierarchy.

    Runs periodically (like sleep) to:
    1. Extract patterns from episodic events
    2. Update semantic knowledge with stable patterns
    3. Propose procedural updates for very stable knowledge
    """

    def __init__(self, memory: FilesystemMemory):
        self.memory = memory

    def consolidate(
        self,
        episodic_to_semantic: bool = True,
        semantic_to_procedural: bool = False,
        max_age_days: int = 7,
    ) -> ConsolidationResult:
        """
        Run consolidation across memory levels.

        Args:
            episodic_to_semantic: Extract patterns from episodes
            semantic_to_procedural: Promote stable knowledge (requires approval)
            max_age_days: How far back to look in episodic memory
        """
        start_time = time.time()
        patterns_found = 0
        patterns_promoted = 0
        knowledge_updated = 0

        if episodic_to_semantic:
            patterns = self._extract_episodic_patterns(max_age_days)
            patterns_found = len(patterns)

            for pattern in patterns:
                if self._should_promote_to_semantic(pattern):
                    self._update_semantic_knowledge(pattern)
                    patterns_promoted += 1
                    knowledge_updated += 1

        if semantic_to_procedural:
            # This is rare and requires human approval
            stable = self._find_stable_semantic_knowledge()
            for knowledge in stable:
                if self._should_promote_to_procedural(knowledge):
                    self._propose_procedural_update(knowledge)

        return ConsolidationResult(
            patterns_found=patterns_found,
            patterns_promoted=patterns_promoted,
            knowledge_updated=knowledge_updated,
            duration_s=time.time() - start_time,
        )

    def _extract_episodic_patterns(self, max_age_days: int) -> list[Pattern]:
        """Extract patterns from recent episodic memory."""
        patterns = []
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        # Analyze RLDS episodes
        episode_patterns = self._analyze_episodes(cutoff_time)
        patterns.extend(episode_patterns)

        # Analyze conversation patterns
        conversation_patterns = self._analyze_conversations(cutoff_time)
        patterns.extend(conversation_patterns)

        # Analyze event patterns
        event_patterns = self._analyze_events(cutoff_time)
        patterns.extend(event_patterns)

        return patterns

    def _analyze_episodes(self, cutoff_time: float) -> list[Pattern]:
        """Analyze RLDS episodes for patterns."""
        patterns = []
        episodes_dir = self.memory.data_dir / "episodic" / "rlds_episodes"

        if not episodes_dir.exists():
            return patterns

        # Collect action sequences from episodes
        action_sequences = Counter()
        successful_strategies = Counter()
        episode_sources = {}

        for ep_dir in episodes_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            metadata_file = ep_dir / "metadata.json"
            steps_file = ep_dir / "steps.jsonl"

            if not metadata_file.exists() or not steps_file.exists():
                continue

            # Check age
            if metadata_file.stat().st_mtime < cutoff_time:
                continue

            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                with open(steps_file) as f:
                    steps = [json.loads(line) for line in f]

                episode_id = metadata.get("episode_id", ep_dir.name)
                success = metadata.get("success", False)

                # Extract action sequence
                actions = [s["action"]["command"] for s in steps if "action" in s]

                # Look for 3-action patterns
                for i in range(len(actions) - 2):
                    seq = tuple(actions[i:i+3])
                    action_sequences[seq] += 1

                    if seq not in episode_sources:
                        episode_sources[seq] = []
                    episode_sources[seq].append(episode_id)

                # Track successful strategies
                if success:
                    level = metadata.get("level_id", "unknown")
                    key = f"{level}:{','.join(actions[:5])}"
                    successful_strategies[key] += 1

            except (json.JSONDecodeError, KeyError):
                continue

        # Convert frequent sequences to patterns
        for seq, count in action_sequences.items():
            if count >= SEMANTIC_FREQUENCY_THRESHOLD:
                patterns.append(Pattern(
                    key=f"action_sequence:{'-'.join(seq)}",
                    description=f"Frequent action sequence: {' → '.join(seq)}",
                    frequency=count,
                    confidence=min(count / 10, 1.0),
                    source_episodes=episode_sources.get(seq, [])[:5],
                    first_seen=cutoff_time,
                    last_seen=time.time(),
                ))

        return patterns

    def _analyze_conversations(self, cutoff_time: float) -> list[Pattern]:
        """Analyze conversation logs for patterns."""
        patterns = []
        conv_dir = self.memory.data_dir / "episodic" / "conversations"

        if not conv_dir.exists():
            return patterns

        # Collect intent patterns
        intent_sequences = Counter()

        for conv_file in conv_dir.glob("*.jsonl"):
            if conv_file.stat().st_mtime < cutoff_time:
                continue

            try:
                with open(conv_file) as f:
                    turns = [json.loads(line) for line in f]

                # Extract user intents
                intents = [
                    t.get("metadata", {}).get("intent", "UNKNOWN")
                    for t in turns
                    if t.get("role") == "user"
                ]

                # Look for 2-intent patterns
                for i in range(len(intents) - 1):
                    seq = tuple(intents[i:i+2])
                    intent_sequences[seq] += 1

            except (json.JSONDecodeError, KeyError):
                continue

        # Convert frequent patterns
        for seq, count in intent_sequences.items():
            if count >= SEMANTIC_FREQUENCY_THRESHOLD:
                patterns.append(Pattern(
                    key=f"intent_pattern:{'-'.join(seq)}",
                    description=f"Users often {seq[0]} then {seq[1]}",
                    frequency=count,
                    confidence=min(count / 10, 1.0),
                    source_episodes=[],
                    first_seen=cutoff_time,
                    last_seen=time.time(),
                ))

        return patterns

    def _analyze_events(self, cutoff_time: float) -> list[Pattern]:
        """Analyze event logs for patterns."""
        patterns = []
        events_file = self.memory.data_dir / "episodic" / "events" / "events.jsonl"

        if not events_file.exists():
            return patterns

        # Collect event type patterns
        event_types = Counter()
        sandbox_denials = Counter()

        try:
            with open(events_file) as f:
                for line in f:
                    event = json.loads(line)

                    if event.get("timestamp", 0) < cutoff_time:
                        continue

                    event_type = event.get("type", "unknown")
                    event_types[event_type] += 1

                    # Track sandbox denials
                    if event.get("sandbox_denied"):
                        action = event.get("action", "unknown")
                        sandbox_denials[action] += 1

        except (json.JSONDecodeError, KeyError):
            pass

        # Create patterns for frequent sandbox denials (learning what NOT to do)
        for action, count in sandbox_denials.items():
            if count >= 2:  # Lower threshold for safety-related patterns
                patterns.append(Pattern(
                    key=f"sandbox_denial:{action}",
                    description=f"Action '{action}' frequently denied by sandbox ({count}x)",
                    frequency=count,
                    confidence=0.9,  # High confidence for safety patterns
                    source_episodes=[],
                    first_seen=cutoff_time,
                    last_seen=time.time(),
                ))

        return patterns

    def _should_promote_to_semantic(self, pattern: Pattern) -> bool:
        """Check if pattern should be promoted to semantic memory."""
        return (
            pattern.frequency >= SEMANTIC_FREQUENCY_THRESHOLD and
            pattern.confidence >= SEMANTIC_CONFIDENCE_THRESHOLD
        )

    def _update_semantic_knowledge(self, pattern: Pattern) -> None:
        """Update semantic knowledge with pattern."""
        # Determine knowledge category
        if pattern.key.startswith("action_sequence:"):
            category = "strategies"
        elif pattern.key.startswith("intent_pattern:"):
            category = "interaction_patterns"
        elif pattern.key.startswith("sandbox_denial:"):
            category = "safety_constraints"
        else:
            category = "general"

        # Build knowledge entry
        content = f"""# {pattern.key}

## Description
{pattern.description}

## Statistics
- Frequency: {pattern.frequency}
- Confidence: {pattern.confidence:.2f}
- First seen: {time.strftime('%Y-%m-%d', time.localtime(pattern.first_seen))}
- Last seen: {time.strftime('%Y-%m-%d', time.localtime(pattern.last_seen))}

## Source Episodes
{chr(10).join(f'- {ep}' for ep in pattern.source_episodes[:5])}

---
*Auto-consolidated from episodic memory*
"""

        self.memory.update_knowledge(
            f"{category}/{pattern.key.replace(':', '_').replace(',', '-')}",
            content,
            source="consolidation"
        )

    def _find_stable_semantic_knowledge(self) -> list[dict]:
        """Find semantic knowledge that's stable enough for procedural promotion."""
        stable = []
        knowledge_dir = self.memory.data_dir / "semantic" / "knowledge"

        if not knowledge_dir.exists():
            return stable

        min_age = time.time() - (PROCEDURAL_AGE_THRESHOLD_DAYS * 24 * 60 * 60)

        for path in knowledge_dir.rglob("*.md"):
            stat = path.stat()

            # Check age
            if stat.st_mtime > min_age:
                continue

            # Check consistency (not modified recently)
            if stat.st_mtime == stat.st_ctime:  # Never modified since creation
                stable.append({
                    "path": str(path),
                    "content": path.read_text(),
                    "age_days": (time.time() - stat.st_ctime) / (24 * 60 * 60),
                })

        return stable

    def _should_promote_to_procedural(self, knowledge: dict) -> bool:
        """Check if knowledge should be promoted to procedural."""
        return knowledge.get("age_days", 0) >= PROCEDURAL_AGE_THRESHOLD_DAYS

    def _propose_procedural_update(self, knowledge: dict) -> None:
        """Propose a procedural update (requires human approval)."""
        proposals_file = self.memory.data_dir / "procedural" / "pending_proposals.json"

        proposals = []
        if proposals_file.exists():
            with open(proposals_file) as f:
                proposals = json.load(f)

        proposals.append({
            "timestamp": time.time(),
            "source_path": knowledge["path"],
            "proposed_action": "promote_to_procedural",
            "status": "pending",
            "content_preview": knowledge["content"][:500],
        })

        with open(proposals_file, "w") as f:
            json.dump(proposals, f, indent=2)


def run_consolidation(
    data_dir: str = "./brain_b_data",
    max_age_days: int = 7,
) -> ConsolidationResult:
    """
    Run memory consolidation.

    This should be called periodically (e.g., on session end, daily, or
    when the robot is idle/charging).
    """
    memory = FilesystemMemory(data_dir)
    consolidator = MemoryConsolidator(memory)

    result = consolidator.consolidate(
        episodic_to_semantic=True,
        semantic_to_procedural=False,  # Disabled by default (requires approval)
        max_age_days=max_age_days,
    )

    print(f"Consolidation complete:")
    print(f"  - Patterns found: {result.patterns_found}")
    print(f"  - Patterns promoted: {result.patterns_promoted}")
    print(f"  - Knowledge updated: {result.knowledge_updated}")
    print(f"  - Duration: {result.duration_s:.2f}s")

    return result


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./brain_b_data"
    run_consolidation(data_dir)
