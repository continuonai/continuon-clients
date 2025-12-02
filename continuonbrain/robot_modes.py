"""
Robot operational modes for ContinuonBrain.
Manages transitions between training, autonomous, and sleep modes.
"""
import time
import json
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


class RobotMode(Enum):
    """Robot operational modes."""
    MANUAL_TRAINING = "manual_training"  # Human teleop for training data
    AUTONOMOUS = "autonomous"  # VLA policy control
    SLEEP_LEARNING = "sleep_learning"  # Self-training on saved memories
    IDLE = "idle"  # Awake but not active
    EMERGENCY_STOP = "emergency_stop"  # Safety stop


@dataclass
class ModeConfig:
    """Configuration for each mode."""
    mode: RobotMode
    timestamp: float
    allow_motion: bool
    record_episodes: bool
    run_inference: bool
    self_train: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RobotModeManager:
    """
    Manages robot operational modes and transitions.
    Handles manual training, autonomous operation, and sleep learning.
    """
    
    def __init__(self, config_dir: str = "/opt/continuonos/brain"):
        self.config_dir = Path(config_dir)
        self.state_file = self.config_dir / ".robot_mode"
        self.current_mode: RobotMode = RobotMode.IDLE
        self.mode_start_time: float = 0
    
    def get_mode_config(self, mode: RobotMode) -> ModeConfig:
        """Get configuration for a specific mode."""
        configs = {
            RobotMode.MANUAL_TRAINING: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=True,
                record_episodes=True,
                run_inference=False,
                self_train=False,
                metadata={"control_source": "human_teleop"}
            ),
            RobotMode.AUTONOMOUS: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=True,
                record_episodes=True,  # Record for continuous learning
                run_inference=True,
                self_train=False,
                metadata={"control_source": "vla_policy"}
            ),
            RobotMode.SLEEP_LEARNING: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=False,
                record_episodes=False,
                run_inference=False,
                self_train=True,
                metadata={
                    "training_type": "offline_replay",
                    "use_gemma": True  # Use Gemma-3 for knowledge
                }
            ),
            RobotMode.IDLE: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=False,
                record_episodes=False,
                run_inference=False,
                self_train=False,
                metadata={}
            ),
            RobotMode.EMERGENCY_STOP: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=False,
                record_episodes=False,
                run_inference=False,
                self_train=False,
                metadata={"reason": "emergency"}
            ),
        }
        
        return configs[mode]
    
    def set_mode(self, new_mode: RobotMode, metadata: Optional[Dict] = None) -> bool:
        """
        Change robot mode.
        
        Args:
            new_mode: Target mode
            metadata: Optional metadata for the mode
        
        Returns:
            True if mode changed successfully
        """
        old_mode = self.current_mode
        
        # Validate transition
        if not self._validate_transition(old_mode, new_mode):
            print(f"❌ Invalid transition: {old_mode.value} → {new_mode.value}")
            return False
        
        # Get mode config
        config = self.get_mode_config(new_mode)
        if metadata:
            config.metadata.update(metadata)
        
        # Execute mode change
        self.current_mode = new_mode
        self.mode_start_time = time.time()
        
        # Save state
        self._save_state(config)
        
        # Print status
        print("=" * 60)
        print(f"🤖 Mode Change: {old_mode.value} → {new_mode.value}")
        print("=" * 60)
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Motion: {'Enabled' if config.allow_motion else 'Disabled'}")
        print(f"Recording: {'ON' if config.record_episodes else 'OFF'}")
        print(f"Inference: {'ON' if config.run_inference else 'OFF'}")
        print(f"Self-Training: {'ON' if config.self_train else 'OFF'}")
        
        if config.metadata:
            print("\nMode Configuration:")
            for key, value in config.metadata.items():
                print(f"  {key}: {value}")
        
        print("=" * 60)
        print()
        
        return True
    
    def _validate_transition(self, from_mode: RobotMode, to_mode: RobotMode) -> bool:
        """Validate mode transition is allowed."""
        # Emergency stop can always be triggered
        if to_mode == RobotMode.EMERGENCY_STOP:
            return True
        
        # Can't leave emergency stop without going through idle
        if from_mode == RobotMode.EMERGENCY_STOP and to_mode != RobotMode.IDLE:
            return False
        
        # Sleep learning can only start from idle
        if to_mode == RobotMode.SLEEP_LEARNING and from_mode != RobotMode.IDLE:
            print("⚠️  Must be idle before sleep learning")
            return False
        
        # All other transitions allowed
        return True
    
    def _save_state(self, config: ModeConfig):
        """Save mode state to file."""
        state = {
            "mode": config.mode.value,  # Convert enum to string
            "timestamp": config.timestamp,
            "config": {
                "mode": config.mode.value,  # Convert enum to string
                "timestamp": config.timestamp,
                "allow_motion": config.allow_motion,
                "record_episodes": config.record_episodes,
                "run_inference": config.run_inference,
                "self_train": config.self_train,
                "metadata": config.metadata
            }
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> Optional[RobotMode]:
        """Load last mode from file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            mode_str = state.get("mode")
            if mode_str:
                return RobotMode(mode_str)
        
        except Exception as e:
            print(f"Error loading mode state: {e}")
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current mode status."""
        config = self.get_mode_config(self.current_mode)
        duration = time.time() - self.mode_start_time
        
        return {
            "mode": self.current_mode.value,
            "duration_seconds": duration,
            "config": asdict(config),
            "timestamp": time.time()
        }
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Trigger emergency stop."""
        print(f"🚨 EMERGENCY STOP: {reason}")
        self.set_mode(RobotMode.EMERGENCY_STOP, {"reason": reason})
    
    def start_manual_training(self):
        """Enter manual training mode for human teleop."""
        print("🎮 Starting manual training mode...")
        print("   Use Flutter app or web UI to control robot")
        print("   All actions will be recorded for training")
        self.set_mode(RobotMode.MANUAL_TRAINING)
    
    def start_autonomous(self):
        """Enter autonomous mode with VLA policy."""
        print("🤖 Starting autonomous mode...")
        print("   Robot will use VLA policy for control")
        print("   Actions still recorded for continuous learning")
        self.set_mode(RobotMode.AUTONOMOUS)
    
    def start_sleep_learning(self, episodes_to_train: Optional[int] = None):
        """
        Enter sleep learning mode.
        Robot self-trains on saved episodes and uses Gemma for knowledge.
        """
        print("💤 Entering sleep learning mode...")
        print("   Robot will self-train on saved memories")
        print("   Using Gemma-3 model for knowledge extraction")
        print("   Motion disabled during learning")
        
        metadata = {"training_type": "offline_replay", "use_gemma": True}
        if episodes_to_train:
            metadata["episodes_to_train"] = episodes_to_train
        
        self.set_mode(RobotMode.SLEEP_LEARNING, metadata)
    
    def return_to_idle(self):
        """Return to idle mode."""
        print("⏸️  Returning to idle...")
        self.set_mode(RobotMode.IDLE)


def main():
    """Test mode manager."""
    manager = RobotModeManager(config_dir="/tmp/mode_test")
    
    print("Testing robot mode transitions...\n")
    
    # Idle → Manual Training
    manager.start_manual_training()
    time.sleep(2)
    
    # Manual Training → Idle
    manager.return_to_idle()
    time.sleep(1)
    
    # Idle → Autonomous
    manager.start_autonomous()
    time.sleep(2)
    
    # Autonomous → Idle
    manager.return_to_idle()
    time.sleep(1)
    
    # Idle → Sleep Learning
    manager.start_sleep_learning(episodes_to_train=16)
    time.sleep(2)
    
    # Emergency stop (can trigger from any mode)
    manager.emergency_stop("Test trigger")
    time.sleep(1)
    
    # Emergency → Idle (required)
    manager.return_to_idle()
    
    print("\n✅ Mode transition test complete!")


if __name__ == "__main__":
    main()
