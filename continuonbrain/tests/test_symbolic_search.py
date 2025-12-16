
import unittest
from continuonbrain.reasoning.tree_search import symbolic_search
from continuonbrain.mamba_brain.world_model import StubSSMWorldModel, WorldModelState
from continuonbrain.reasoning.arm_state_codec import ArmGoal

class TestSymbolicSearch(unittest.TestCase):
    def test_search_finds_plan(self):
        # Setup clean state
        start = WorldModelState(joint_pos=[0.0] * 6)
        # Goal is reachable: just move joint 0 by 0.2
        goal = ArmGoal(target_joint_pos=[0.2] + [0.0] * 5)
        
        # Use stub model (deterministic)
        model = StubSSMWorldModel(joint_limit=1.0)
        
        # Run search
        action = symbolic_search(start, goal, model, steps=5)
        
        print(f"Found Action: {action}")
        
        self.assertIsNotNone(action, "Search should find a plan for simple goal")
        self.assertEqual(len(action), 6)
        # Verify it tries to move towards goal (positive delta on joint 0)
        self.assertGreater(action[0], 0.0)

if __name__ == '__main__':
    unittest.main()
