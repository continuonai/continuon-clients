import json
from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

class RobotControllerMixin:
    """
    Mixin for processing Robot Control API requests.
    """

    @require_role(UserRole.CONSUMER) # Allows owners and creators
    def handle_drive(self, body: str):
        brain_service = self.server.brain_service
        data = json.loads(body)
        steering = float(data.get("steering", 0.0))
        throttle = float(data.get("throttle", 0.0))
        
        if brain_service.drivetrain:
            brain_service.drivetrain.apply_drive(steering, throttle)
            self.send_json({"success": True})
        else:
            self.send_json({"success": False, "message": "No drivetrain"})

    @require_role(UserRole.CONSUMER)
    def handle_joints(self, body: str):
        brain_service = self.server.brain_service
        data = json.loads(body)
        joint_idx = data.get("joint_index")
        val = data.get("value")
        
        if brain_service.arm and joint_idx is not None:
            # ROUTE THROUGH SAFETY KERNEL
            joint_map = {0: "base", 1: "shoulder", 2: "elbow", 3: "wrist", 4: "gripper"}
            joint_name = joint_map.get(joint_idx, f"joint_{joint_idx}")
            
            res = brain_service.safety_client.send_command("move_joints", {"joints": {joint_name: float(val)}})
            
            if res.get("status") != "ok":
                    self.send_json({"success": False, "message": f"Safety Kernel Blocked: {res.get('reason')}"})
                    return

            # Apply safe action
            safe_val = res.get("args", {}).get("joints", {}).get(joint_name, val)
            
            # Get current state first
            current = brain_service.arm.get_normalized_state()
            # Determine target
            target = list(current)
            if 0 <= joint_idx < 6:
                target[joint_idx] = float(safe_val)
                brain_service.arm.set_normalized_action(target)
                self.send_json({"success": True, "clipping": res.get("safety_level") == 1})
            else:
                self.send_json({"success": False, "message": "Invalid joint index"})
        else:
            self.send_json({"success": False, "message": "No arm or invalid data"})

    @require_role(UserRole.CONSUMER)
    def handle_mode_set(self, target_mode: str):
        # Delegate to self._set_mode which is existing method in server.py
        # But we want to protect it.
        # Calling the original method from here.
        result = self._set_mode(target_mode) 
        self.send_json(result, status=200 if result.get("success") else 400)

    @require_role(UserRole.CONSUMER)
    def handle_safety_action(self, action: str):
        target = "emergency_stop" if action == "hold" else "idle"
        result = self._set_mode(target)
        self.send_json(result, status=200 if result.get("success") else 400)
