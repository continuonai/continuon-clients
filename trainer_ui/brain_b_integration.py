"""
Brain B Integration for Trainer UI

Connects arm commands to Brain B for:
- Action validation (safety checks via guardrails)
- Teaching integration (record arm movements as behaviors)
- Conversation handling (natural language arm control)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add brain_b to path
sys.path.insert(0, str(Path(__file__).parent.parent / "brain_b"))

try:
    from actor_runtime import ActorRuntime
    from conversation.handler import ConversationHandler
    HAS_BRAIN_B = True
except ImportError:
    HAS_BRAIN_B = False
    print("Brain B not available - arm validation disabled")


class BrainBIntegration:
    """
    Integrates Brain B with Trainer UI for intelligent arm control.

    Features:
    - Validates arm actions against safety guardrails
    - Records arm movements for Brain B teaching
    - Provides natural language arm control via ConversationHandler
    """

    # Arm action safety limits
    ARM_LIMITS = {
        "joint_speed": 0.1,  # Max change per update
        "gripper_speed": 0.2,  # Max gripper change per update
        "joint_min": -1.0,
        "joint_max": 1.0,
        "gripper_min": 0.0,
        "gripper_max": 1.0,
    }

    def __init__(self, data_path: str = "./brain_b_data"):
        """
        Initialize Brain B integration.

        Args:
            data_path: Path for Brain B data storage
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.runtime: Optional[ActorRuntime] = None
        self.handler: Optional[ConversationHandler] = None
        self.last_arm_state: Dict[str, Dict[str, float]] = {}

        if HAS_BRAIN_B:
            self._initialize_brain_b()

    def _initialize_brain_b(self):
        """Initialize Brain B runtime and handler."""
        try:
            self.runtime = ActorRuntime(
                data_path=str(self.data_path),
                auto_restore=True,
            )

            # Create a dummy executor for arm actions
            def arm_executor(action: dict):
                # This is called when Brain B wants to execute an arm action
                # The actual execution happens in the server
                pass

            self.handler = ConversationHandler(
                runtime=self.runtime,
                executor=arm_executor,
                use_ai=True,
                prefer_gemini=True,
            )

            print(f"[BrainB] Initialized with data path: {self.data_path}")

        except Exception as e:
            print(f"[BrainB] Failed to initialize: {e}")
            self.runtime = None
            self.handler = None

    @property
    def is_available(self) -> bool:
        """Check if Brain B integration is available."""
        return HAS_BRAIN_B and self.runtime is not None

    def validate_arm_action(
        self,
        arm_id: str,
        joint: int,
        value: float,
        current_state: Optional[Dict] = None,
    ) -> Tuple[bool, str, float]:
        """
        Validate an arm joint action against safety limits.

        Args:
            arm_id: Which arm (arm_0 or arm_1)
            joint: Joint index (0-4)
            value: Target value (-1.0 to 1.0)
            current_state: Current arm state

        Returns:
            Tuple of (is_valid, message, adjusted_value)
        """
        # Clamp to valid range
        clamped = max(self.ARM_LIMITS["joint_min"], min(self.ARM_LIMITS["joint_max"], value))

        # Check rate limiting if we have previous state
        if current_state and arm_id in self.last_arm_state:
            prev_state = self.last_arm_state[arm_id]
            prev_value = prev_state.get(f"joint_{joint}", 0.0)
            delta = abs(clamped - prev_value)

            if delta > self.ARM_LIMITS["joint_speed"]:
                # Rate limit - move towards target but not too fast
                direction = 1 if clamped > prev_value else -1
                clamped = prev_value + (direction * self.ARM_LIMITS["joint_speed"])
                return True, f"Rate limited (delta={delta:.2f})", clamped

        # Update last state
        if arm_id not in self.last_arm_state:
            self.last_arm_state[arm_id] = {}
        self.last_arm_state[arm_id][f"joint_{joint}"] = clamped

        return True, "OK", clamped

    def validate_gripper_action(
        self,
        arm_id: str,
        value: float,
        current_state: Optional[Dict] = None,
    ) -> Tuple[bool, str, float]:
        """
        Validate a gripper action.

        Args:
            arm_id: Which arm
            value: Target value (0.0 to 1.0)
            current_state: Current arm state

        Returns:
            Tuple of (is_valid, message, adjusted_value)
        """
        # Clamp to valid range
        clamped = max(self.ARM_LIMITS["gripper_min"], min(self.ARM_LIMITS["gripper_max"], value))

        # Rate limiting
        if current_state and arm_id in self.last_arm_state:
            prev_value = self.last_arm_state[arm_id].get("gripper", 0.0)
            delta = abs(clamped - prev_value)

            if delta > self.ARM_LIMITS["gripper_speed"]:
                direction = 1 if clamped > prev_value else -1
                clamped = prev_value + (direction * self.ARM_LIMITS["gripper_speed"])

        # Update state
        if arm_id not in self.last_arm_state:
            self.last_arm_state[arm_id] = {}
        self.last_arm_state[arm_id]["gripper"] = clamped

        return True, "OK", clamped

    def record_arm_action(
        self,
        action_type: str,
        arm_id: str,
        **kwargs,
    ):
        """
        Record an arm action for Brain B teaching.

        Args:
            action_type: "arm" or "gripper"
            arm_id: Which arm
            **kwargs: Additional action parameters
        """
        if not self.is_available:
            return

        action = {
            "type": f"{action_type}_{arm_id}",
            "arm_id": arm_id,
            **kwargs,
        }

        # If teaching mode is active in Brain B, record the action
        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action(action)

    def process_natural_language(self, text: str) -> Tuple[str, Optional[Dict]]:
        """
        Process natural language input for arm control.

        Args:
            text: Natural language input

        Returns:
            Tuple of (response_text, arm_action or None)
        """
        if not self.is_available or not self.handler:
            return ("Brain B not available", None)

        response = self.handler.handle(text)

        # Check if the response includes an arm action
        arm_action = None
        if response.action_taken and response.action_type:
            # Map Brain B actions to arm actions
            action_map = {
                "arm_up": {"type": "arm", "joint": 1, "value": 0.5},
                "arm_down": {"type": "arm", "joint": 1, "value": -0.5},
                "grab": {"type": "gripper", "value": 1.0},
                "release": {"type": "gripper", "value": 0.0},
                "wave": {"type": "pose", "name": "wave_up"},
            }
            arm_action = action_map.get(response.action_type)

        return (response.text, arm_action)

    def start_teaching(self, name: str) -> str:
        """Start teaching a new arm behavior."""
        if not self.is_available:
            return "Brain B not available"
        return self.runtime.teach(name)

    def stop_teaching(self) -> str:
        """Stop teaching and save the behavior."""
        if not self.is_available:
            return "Brain B not available"
        return self.runtime.done_teaching()

    def invoke_behavior(self, name: str, executor) -> str:
        """Execute a learned arm behavior."""
        if not self.is_available:
            return "Brain B not available"
        return self.runtime.invoke(name, executor)

    def list_behaviors(self) -> list:
        """List learned arm behaviors."""
        if not self.is_available:
            return []
        return self.runtime.teaching.list_behaviors()

    def shutdown(self):
        """Clean shutdown."""
        if self.runtime:
            self.runtime.shutdown()
