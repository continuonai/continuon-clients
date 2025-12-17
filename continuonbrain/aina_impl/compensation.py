
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

class SensorFusionModule:
    """
    Compensates for AINA system weaknesses:
    1. Occlusion (Hand blocking object) -> Uses temporal memory (Object Permanence).
    2. Depth Ambiguity -> Fuses 'Inferred' depth with 'Active' OAK-D depth if available.
    """
    def __init__(self, confidence_threshold=0.6, memory_decay=0.99):
        self.confidence_threshold = confidence_threshold
        self.memory_decay = memory_decay
        
        # Working Memory of Object State [x, y, z]
        self.last_known_pos = None
        self.last_update_time = 0
        
    def fuse_inputs(self, 
                    inferred_pos: np.ndarray, 
                    inferred_conf: float, 
                    active_depth_pos: np.ndarray = None,
                    timestamp: float = 0.0):
        """
        Fuse inputs to get robust object position.
        
        Args:
            inferred_pos: [x,y,z] from Passive Stereo / FoundationModels
            inferred_conf: Confidence score (0.0 - 1.0) from Tracker
            active_depth_pos: [x,y,z] from OAK-D Active Stereo (Ground Truth-ish)
            timestamp: Current time
            
        Returns:
            fused_pos: [x,y,z]
            status: str (Description of decision)
        """
        
        # 1. Handle Active Depth (Ground Truth Correction)
        # If we have active depth, we trust it more than passive inference, 
        # BUT we must cross-reference it with the visual tracker to ensure
        # we aren't getting depth of the *hand* blocking the object.
        # Simple heuristic: If active depth is significantly closer than inferred 
        # depth AND inferred confidence is low, it might be occlusion.
        
        current_est = inferred_pos
        
        # Fusion weights
        w_inferred = 1.0
        w_active = 0.0
        
        if active_depth_pos is not None:
            # Check for consistency
            dist = np.linalg.norm(inferred_pos - active_depth_pos)
            
            if dist < 0.1: # 10cm agreement
                # High agreement, trust active more for precision
                w_inferred = 0.2
                w_active = 0.8
                current_est = (inferred_pos * w_inferred + active_depth_pos * w_active)
            else:
                # Disagreement. 
                # Case A: Occlusion? Active sees hand (close), Inferred sees nothing/noise.
                # Case B: Inferred is just wrong.
                # We default to Active if available unless told otherwise, 
                # but applied to AINA, we usually only get 2D + Depth map.
                # Let's simple-average for this implementation.
                current_est = (inferred_pos + active_depth_pos) / 2.0

        # 2. Handle Occlusion (Temporal Filtering)
        if inferred_conf < self.confidence_threshold:
            # Low confidence! Potential Occlusion.
            if self.last_known_pos is not None:
                # Retrieve from Memory (Object Permanence)
                # We could apply decay or velocity extrapolation here.
                # For now, simplistic "Latch" memory.
                logger.warning(f"Occlusion detected (Conf {inferred_conf:.2f}). Using Memory.")
                return self.last_known_pos, "MEMORY_RECALL"
            else:
                # No memory, forced to use noisy input
                return current_est, "LOW_CONFIDENCE_LIVE"
        
        # 3. High Confidence Update
        # Update Memory
        self.last_known_pos = current_est
        self.last_update_time = timestamp
        
        return current_est, "LIVE_FUSION"

    def reset(self):
        self.last_known_pos = None
