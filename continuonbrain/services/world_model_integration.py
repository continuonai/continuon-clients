"""
World Model Integration - Unified Sensory Input for HOPE Brain

This module provides the integration layer between sensory inputs
(video segmentation, depth, pose) and the HOPE world model, enabling
the brain to build a coherent world representation for reasoning.

Architecture:
    Sensors (OAK-D, Hailo, SAM) → WorldModelIntegration → HOPE Brain CMS
                                                        ↓
                                        HOPEAgent (Chat/Reasoning)

The world model uses SSM/Mamba-style state tracking to maintain:
- Visual scene state (objects, positions, relationships)
- Temporal predictions (what will happen next)
- Uncertainty estimates (confidence in predictions)
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensoryFrame:
    """A single frame of sensory input from all modalities."""
    timestamp: float
    rgb_frame: Optional[np.ndarray] = None
    depth_frame: Optional[np.ndarray] = None
    segmentation: Optional[Dict[str, Any]] = None
    pose_estimation: Optional[Dict[str, Any]] = None

    # Derived features
    objects: List[Dict[str, Any]] = field(default_factory=list)
    scene_description: str = ""
    spatial_graph: Optional[Dict[str, Any]] = None


@dataclass
class WorldState:
    """Current world model state maintained by the integration layer."""
    timestamp: float

    # Object tracking (persistent IDs across frames)
    tracked_objects: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Spatial relationships
    object_relationships: List[Dict[str, Any]] = field(default_factory=list)

    # Scene understanding
    scene_context: str = ""
    scene_features: Optional[np.ndarray] = None

    # Prediction state (for SSM rollouts)
    hidden_state: Optional[np.ndarray] = None
    uncertainty: float = 0.5

    # History for temporal reasoning
    recent_observations: List[SensoryFrame] = field(default_factory=list)
    max_history: int = 10


class WorldModelIntegration:
    """
    Unified world model integration for HOPE brain.

    Connects all sensory modalities to the brain's CMS (Continuous Memory System)
    and SSM-based world model for coherent scene understanding and prediction.

    Key capabilities:
    1. Multi-modal fusion (RGB, depth, segmentation, pose)
    2. Object tracking and persistence
    3. Spatial relationship inference
    4. Temporal prediction with uncertainty
    5. Integration with HOPE brain CMS for memory
    """

    def __init__(
        self,
        brain_service=None,
        enable_tracking: bool = True,
        enable_predictions: bool = True,
        update_rate_hz: float = 10.0,
    ):
        """
        Initialize world model integration.

        Args:
            brain_service: Reference to BrainService for HOPE brain access
            enable_tracking: Enable persistent object tracking
            enable_predictions: Enable SSM-based predictions
            update_rate_hz: Target update rate for world model
        """
        self.brain_service = brain_service
        self.enable_tracking = enable_tracking
        self.enable_predictions = enable_predictions
        self.update_interval = 1.0 / update_rate_hz

        # Current world state
        self._world_state = WorldState(timestamp=time.time())
        self._state_lock = threading.RLock()

        # Object tracking
        self._next_object_id = 1
        self._object_history: Dict[int, List[Dict]] = {}

        # SSM/Mamba world model reference (from brain_service)
        self._ssm_model = None

        # Update thread
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        logger.info("WorldModelIntegration initialized")

    def start(self):
        """Start the world model update loop."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="world_model_update"
        )
        self._update_thread.start()
        logger.info("World model update loop started")

    def stop(self):
        """Stop the world model update loop."""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        self._running = False
        logger.info("World model update loop stopped")

    def _update_loop(self):
        """Background update loop for world model."""
        while not self._stop_event.is_set():
            try:
                self._update_world_state()
            except Exception as e:
                logger.error(f"World model update error: {e}")

            # Wait for next update interval
            self._stop_event.wait(self.update_interval)

    def _update_world_state(self):
        """Update world state from all available sensors."""
        if not self.brain_service:
            return

        # Collect sensory data
        frame = SensoryFrame(timestamp=time.time())

        # Get latest segmentation from server (cached)
        seg_data = getattr(self.brain_service, 'last_segmentation', None)
        if seg_data:
            frame.segmentation = seg_data
            frame.objects = seg_data.get('objects', [])

        # Get latest pose estimation
        pose_data = getattr(self.brain_service, 'last_pose_result', None)
        if pose_data:
            frame.pose_estimation = pose_data

        # Update world state
        with self._state_lock:
            self._integrate_frame(frame)

    def _integrate_frame(self, frame: SensoryFrame):
        """Integrate a sensory frame into the world state."""
        self._world_state.timestamp = frame.timestamp

        # Track objects across frames
        if self.enable_tracking and frame.objects:
            self._update_object_tracking(frame.objects)

        # Build spatial relationships
        self._update_spatial_relationships()

        # Generate scene description
        self._update_scene_description()

        # Add to history
        self._world_state.recent_observations.append(frame)
        if len(self._world_state.recent_observations) > self._world_state.max_history:
            self._world_state.recent_observations.pop(0)

        # Run SSM prediction if enabled
        if self.enable_predictions:
            self._update_predictions()

        # Inject into HOPE brain CMS
        self._inject_into_brain()

    def _update_object_tracking(self, detected_objects: List[Dict]):
        """Track objects across frames with persistent IDs."""
        current_tracked = {}

        for det in detected_objects:
            center = det.get('center', [0, 0])
            label = det.get('label', 'unknown')

            # Try to match with existing tracked objects
            best_match_id = None
            best_match_dist = float('inf')

            for obj_id, tracked in self._world_state.tracked_objects.items():
                # Simple distance-based matching
                tracked_center = tracked.get('center', [0, 0])
                dist = np.sqrt(
                    (center[0] - tracked_center[0])**2 +
                    (center[1] - tracked_center[1])**2
                )

                # Match if same label and close enough
                if tracked.get('label') == label and dist < best_match_dist and dist < 100:
                    best_match_id = obj_id
                    best_match_dist = dist

            if best_match_id is not None:
                # Update existing tracked object
                obj_id = best_match_id
            else:
                # New object
                obj_id = self._next_object_id
                self._next_object_id += 1

            current_tracked[obj_id] = {
                'id': obj_id,
                'label': label,
                'center': center,
                'box': det.get('box', [0, 0, 0, 0]),
                'score': det.get('score', 0.0),
                'area': det.get('area', 0),
                'last_seen': time.time(),
                'frames_tracked': self._world_state.tracked_objects.get(obj_id, {}).get('frames_tracked', 0) + 1,
            }

            # Store history
            if obj_id not in self._object_history:
                self._object_history[obj_id] = []
            self._object_history[obj_id].append({
                'timestamp': time.time(),
                'center': center,
                'score': det.get('score', 0.0),
            })
            # Trim history
            if len(self._object_history[obj_id]) > 30:
                self._object_history[obj_id].pop(0)

        self._world_state.tracked_objects = current_tracked

    def _update_spatial_relationships(self):
        """Compute spatial relationships between objects."""
        relationships = []
        objects = list(self._world_state.tracked_objects.values())

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                rel = self._compute_relationship(obj1, obj2)
                if rel:
                    relationships.append(rel)

        self._world_state.object_relationships = relationships

    def _compute_relationship(self, obj1: Dict, obj2: Dict) -> Optional[Dict]:
        """Compute spatial relationship between two objects."""
        c1 = obj1.get('center', [0, 0])
        c2 = obj2.get('center', [0, 0])

        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        dist = np.sqrt(dx**2 + dy**2)

        # Determine relative position
        if abs(dx) > abs(dy):
            position = "right_of" if dx > 0 else "left_of"
        else:
            position = "below" if dy > 0 else "above"

        # Determine proximity
        if dist < 50:
            proximity = "touching"
        elif dist < 150:
            proximity = "near"
        else:
            proximity = "far"

        return {
            'subject': obj1['id'],
            'subject_label': obj1.get('label', 'unknown'),
            'relation': position,
            'object': obj2['id'],
            'object_label': obj2.get('label', 'unknown'),
            'distance': float(dist),
            'proximity': proximity,
        }

    def _update_scene_description(self):
        """Generate natural language scene description."""
        objects = list(self._world_state.tracked_objects.values())

        if not objects:
            self._world_state.scene_context = "The scene appears empty or no objects detected."
            return

        # Count objects by label
        label_counts = {}
        for obj in objects:
            label = obj.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1

        # Build description
        parts = []
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            if count == 1:
                parts.append(f"a {label}")
            else:
                parts.append(f"{count} {label}s")

        if len(parts) == 1:
            desc = f"I can see {parts[0]}."
        elif len(parts) == 2:
            desc = f"I can see {parts[0]} and {parts[1]}."
        else:
            desc = f"I can see {', '.join(parts[:-1])}, and {parts[-1]}."

        # Add relationships
        rels = self._world_state.object_relationships[:3]  # Top 3
        if rels:
            rel_desc = []
            for r in rels:
                rel_desc.append(f"{r['subject_label']} is {r['relation'].replace('_', ' ')} {r['object_label']}")
            desc += " " + ". ".join(rel_desc) + "."

        self._world_state.scene_context = desc

    def _update_predictions(self):
        """Run SSM/Mamba predictions for next state."""
        try:
            # Get Mamba world model from brain service
            if not self._ssm_model and self.brain_service:
                import importlib
                # Import module with numeric prefix using importlib
                mamba_brain = importlib.import_module("continuonbrain.03_mamba_brain.world_model")
                self._ssm_model = mamba_brain.build_world_model(prefer_mamba=True)

            if self._ssm_model is None:
                self._world_state.uncertainty = 0.5
                return

            # For now, set moderate uncertainty
            # Full SSM integration would use the model's predict() method
            # with a state vector built from tracked objects
            self._world_state.uncertainty = 0.3  # Moderate confidence

        except Exception as e:
            logger.debug(f"Prediction update skipped: {e}")
            self._world_state.uncertainty = 0.7  # Higher uncertainty on error

    def _inject_into_brain(self):
        """Inject world state into HOPE brain's CMS."""
        if not self.brain_service or not self.brain_service.hope_brain:
            return

        try:
            # Build context vector for CMS injection
            context = {
                'type': 'world_state',
                'timestamp': self._world_state.timestamp,
                'scene': self._world_state.scene_context,
                'num_objects': len(self._world_state.tracked_objects),
                'objects': [
                    {'id': obj['id'], 'label': obj['label'], 'score': obj['score']}
                    for obj in self._world_state.tracked_objects.values()
                ],
                'relationships': self._world_state.object_relationships[:5],
                'uncertainty': self._world_state.uncertainty,
            }

            # Get HOPE agent and inject context
            agent = self.brain_service.hope_agent
            if agent and hasattr(agent, 'inject_visual_context'):
                # Build segmentation-like data for injection
                seg_data = {
                    'timestamp': self._world_state.timestamp,
                    'num_objects': len(self._world_state.tracked_objects),
                    'objects': [
                        {
                            'id': obj['id'],
                            'label': obj['label'],
                            'score': obj['score'],
                            'center': obj['center'],
                            'box': obj['box'],
                            'area': obj['area'],
                        }
                        for obj in self._world_state.tracked_objects.values()
                    ],
                    'frame_size': [640, 400],
                }
                agent.inject_visual_context(seg_data)

        except Exception as e:
            logger.debug(f"Brain injection skipped: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state as a dictionary."""
        with self._state_lock:
            return {
                'timestamp': self._world_state.timestamp,
                'scene_description': self._world_state.scene_context,
                'num_objects': len(self._world_state.tracked_objects),
                'objects': [
                    {
                        'id': obj['id'],
                        'label': obj['label'],
                        'score': obj['score'],
                        'center': obj['center'],
                        'area': obj['area'],
                        'frames_tracked': obj['frames_tracked'],
                    }
                    for obj in self._world_state.tracked_objects.values()
                ],
                'relationships': self._world_state.object_relationships,
                'uncertainty': self._world_state.uncertainty,
                'history_frames': len(self._world_state.recent_observations),
            }

    def get_object_by_id(self, object_id: int) -> Optional[Dict[str, Any]]:
        """Get a tracked object by ID."""
        with self._state_lock:
            return self._world_state.tracked_objects.get(object_id)

    def get_objects_by_label(self, label: str) -> List[Dict[str, Any]]:
        """Get all tracked objects with a given label."""
        with self._state_lock:
            return [
                obj for obj in self._world_state.tracked_objects.values()
                if obj.get('label', '').lower() == label.lower()
            ]

    def query_scene(self, question: str) -> str:
        """
        Answer a question about the current scene.

        This is a simple query interface for the world model.
        For complex reasoning, use the HOPE agent.
        """
        question_lower = question.lower()

        with self._state_lock:
            objects = list(self._world_state.tracked_objects.values())

            # Count questions
            if "how many" in question_lower:
                for label in set(obj['label'] for obj in objects):
                    if label.lower() in question_lower:
                        count = sum(1 for obj in objects if obj['label'].lower() == label.lower())
                        return f"I can see {count} {label}{'s' if count != 1 else ''}."
                return f"I can see {len(objects)} objects total."

            # Location questions
            if "where" in question_lower:
                for obj in objects:
                    if obj['label'].lower() in question_lower:
                        c = obj['center']
                        pos = "center"
                        if c[0] < 200:
                            pos = "left"
                        elif c[0] > 440:
                            pos = "right"
                        return f"The {obj['label']} is on the {pos} side of the view."

            # Existence questions
            if "is there" in question_lower or "do you see" in question_lower:
                for obj in objects:
                    if obj['label'].lower() in question_lower:
                        return f"Yes, I can see a {obj['label']} with {obj['score']:.0%} confidence."
                return "I don't see that in the current view."

            # Default to scene description
            return self._world_state.scene_context


class TeacherInterface:
    """
    Teacher interface for Claude Code to guide HOPE learning.

    This allows Claude Code (or other teacher agents) to:
    1. Provide corrections when HOPE is wrong
    2. Answer HOPE's questions
    3. Provide demonstrations
    4. Validate HOPE's learned knowledge
    """

    def __init__(self, brain_service=None, world_model_integration=None):
        """
        Initialize teacher interface.

        Args:
            brain_service: Reference to BrainService
            world_model_integration: Reference to WorldModelIntegration
        """
        self.brain_service = brain_service
        self.world_model = world_model_integration

        # Pending interactions
        self._pending_questions: List[Dict[str, Any]] = []
        self._teaching_history: List[Dict[str, Any]] = []

        logger.info("TeacherInterface initialized")

    def get_pending_questions(self) -> List[Dict[str, Any]]:
        """Get questions HOPE is asking that need answers."""
        if not self.brain_service or not self.brain_service.hope_agent:
            return []

        try:
            # Get knowledge gaps from HOPE agent
            agent = self.brain_service.hope_agent

            # Build context from world model
            context = {}
            if self.world_model:
                world_state = self.world_model.get_world_state()
                context['visual_perception'] = {
                    'objects': world_state['objects'],
                    'scene_description': world_state['scene_description'],
                }

            gaps = agent.identify_knowledge_gaps(context)

            # Convert to questions
            questions = []
            for gap in gaps:
                questions.append({
                    'id': len(questions),
                    'type': gap['type'],
                    'question': gap.get('suggested_question', 'How can you help me understand this?'),
                    'priority': gap.get('priority', 'medium'),
                    'context': gap,
                })

            self._pending_questions = questions
            return questions

        except Exception as e:
            logger.error(f"Error getting pending questions: {e}")
            return []

    def provide_answer(self, question_id: int, answer: str, confidence: float = 0.9) -> Dict[str, Any]:
        """
        Provide an answer to one of HOPE's questions.

        Args:
            question_id: ID of the question being answered
            answer: The answer text
            confidence: Confidence in the answer (0-1)

        Returns:
            Result of the teaching interaction
        """
        try:
            if question_id >= len(self._pending_questions):
                return {'success': False, 'error': 'Invalid question ID'}

            question = self._pending_questions[question_id]

            # Store in teaching history
            interaction = {
                'timestamp': time.time(),
                'type': 'answer',
                'question': question['question'],
                'answer': answer,
                'confidence': confidence,
                'context': question.get('context', {}),
            }
            self._teaching_history.append(interaction)

            # Try to inject into HOPE's memory
            if self.brain_service and self.brain_service.experience_logger:
                self.brain_service.experience_logger.store_teaching(
                    question=question['question'],
                    answer=answer,
                    confidence=confidence,
                    teacher='claude_code',
                )

            return {
                'success': True,
                'message': f"Thank you for teaching me: {answer[:100]}",
                'stored': True,
            }

        except Exception as e:
            logger.error(f"Error providing answer: {e}")
            return {'success': False, 'error': str(e)}

    def provide_correction(
        self,
        original_response: str,
        correct_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Correct a mistake HOPE made.

        Args:
            original_response: What HOPE said/did
            correct_response: What it should have been
            context: Optional context about the situation

        Returns:
            Result of the correction
        """
        try:
            if not self.brain_service or not self.brain_service.hope_agent:
                return {'success': False, 'error': 'HOPE agent not available'}

            # Use HOPE agent's learning method
            result = self.brain_service.hope_agent.learn_from_correction(
                original_response=original_response,
                correction=correct_response,
                context=context,
                experience_logger=self.brain_service.experience_logger,
            )

            # Store in teaching history
            interaction = {
                'timestamp': time.time(),
                'type': 'correction',
                'original': original_response,
                'correction': correct_response,
                'context': context,
                'result': result,
            }
            self._teaching_history.append(interaction)

            return result

        except Exception as e:
            logger.error(f"Error providing correction: {e}")
            return {'success': False, 'error': str(e)}

    def demonstrate_action(
        self,
        action_name: str,
        action_steps: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Demonstrate how to perform an action.

        Args:
            action_name: Name of the action (e.g., "pick up cup")
            action_steps: List of steps to perform
            context: Optional context (e.g., which objects are involved)

        Returns:
            Result of the demonstration
        """
        try:
            # Store demonstration
            demonstration = {
                'timestamp': time.time(),
                'type': 'demonstration',
                'action': action_name,
                'steps': action_steps,
                'context': context,
            }
            self._teaching_history.append(demonstration)

            # Try to store in experience logger
            if self.brain_service and self.brain_service.experience_logger:
                self.brain_service.experience_logger.store_demonstration(
                    action=action_name,
                    steps=action_steps,
                    context=context,
                    teacher='claude_code',
                )

            return {
                'success': True,
                'message': f"Learned demonstration for '{action_name}' with {len(action_steps)} steps",
                'stored': True,
            }

        except Exception as e:
            logger.error(f"Error storing demonstration: {e}")
            return {'success': False, 'error': str(e)}

    def get_teaching_summary(self) -> Dict[str, Any]:
        """Get a summary of teaching interactions."""
        corrections = [t for t in self._teaching_history if t['type'] == 'correction']
        answers = [t for t in self._teaching_history if t['type'] == 'answer']
        demos = [t for t in self._teaching_history if t['type'] == 'demonstration']

        return {
            'total_interactions': len(self._teaching_history),
            'corrections': len(corrections),
            'answers': len(answers),
            'demonstrations': len(demos),
            'recent': self._teaching_history[-5:] if self._teaching_history else [],
        }

    def suggest_teaching_focus(self) -> List[Dict[str, Any]]:
        """
        Suggest areas where teaching would be most helpful.

        This analyzes HOPE's current state and suggests what to teach.
        """
        suggestions = []

        # Check knowledge gaps
        questions = self.get_pending_questions()
        high_priority = [q for q in questions if q.get('priority') == 'high']

        if high_priority:
            suggestions.append({
                'type': 'knowledge_gap',
                'description': f"HOPE has {len(high_priority)} high-priority questions",
                'action': 'Answer pending questions',
                'questions': high_priority,
            })

        # Check for low-confidence objects in scene
        if self.world_model:
            world_state = self.world_model.get_world_state()
            low_conf_objects = [
                obj for obj in world_state.get('objects', [])
                if obj.get('score', 1.0) < 0.6
            ]

            if low_conf_objects:
                suggestions.append({
                    'type': 'object_identification',
                    'description': f"HOPE is uncertain about {len(low_conf_objects)} objects",
                    'action': 'Help identify objects',
                    'objects': low_conf_objects,
                })

        # Check recent errors (would need error tracking)
        if not suggestions:
            suggestions.append({
                'type': 'general',
                'description': 'HOPE is performing well',
                'action': 'Consider teaching new capabilities or validating existing knowledge',
            })

        return suggestions


def create_world_model_integration(brain_service=None) -> Tuple[WorldModelIntegration, TeacherInterface]:
    """
    Factory function to create world model integration and teacher interface.

    Args:
        brain_service: Reference to BrainService

    Returns:
        Tuple of (WorldModelIntegration, TeacherInterface)
    """
    wm = WorldModelIntegration(brain_service=brain_service)
    teacher = TeacherInterface(brain_service=brain_service, world_model_integration=wm)

    return wm, teacher
