
import numpy as np
import time
from continuonbrain.services.agent_hope import HOPEAgent
from continuonbrain.hope_impl.config import HOPEConfig
from continuonbrain.hope_impl.brain import HOPEBrain

def test_confidence_accuracy():
    print("--- Testing HOPE Confidence Accuracy ---")
    
    # 1. Initialize HOPE Brain (Simulated)
    try:
        config = HOPEConfig(num_columns=1)
        brain = HOPEBrain(config, obs_dim=10, action_dim=4, output_dim=10)
        agent = HOPEAgent(brain, None) # No experience logger needed for confidence test
        print("HOPE Agent initialized.")
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return

    # 2. Simulate Training (Familiar Pattern)
    print("\n[Scenario 1] Familiar Pattern")
    pattern_A = np.ones(10) * 0.5
    
    # Train heavily on pattern A
    print("Training on Pattern A...")
    a_prev = np.zeros(4)
    for _ in range(50):
        # Manually invoke components if possible, or just rely on internal state if exposed
        # Since agent_hope calculates confidence based on energy/stability,
        # we need to simulate brain activity.
        # However, HOPEAgent wraps the brain.
        # Let's interact with brain directly to 'train' it.
        brain.step(x_obs=pattern_A, a_prev=a_prev, r_t=1.0, perform_param_update=True)
    
    # Check confidence for Pattern A
    # Calling can_answer assesses confidence of current brain state (which just processed Pattern A)
    _, conf_A = agent.can_answer("generic query")
    metrics_A = brain.columns[brain.active_column_idx].stability_monitor.get_metrics()
    print(f"Confidence (Familiar): {conf_A:.4f}")
    print(f"Metrics: {metrics_A}")
    
    if conf_A > 0.6:
        print("PASS: Confidence high for familiar pattern.")
    else:
        print("FAIL: Confidence too low for familiar pattern.")

    # 3. novel Pattern
    print("\n[Scenario 2] Novel Pattern")
    pattern_B = np.zeros(10) # Different input
    
    # Step brain with novel pattern to update state (but don't learn it yet)
    brain.step(x_obs=pattern_B, a_prev=a_prev, r_t=0.0, perform_param_update=False)
    
    _, conf_B = agent.can_answer("generic query")
    metrics_B = brain.columns[brain.active_column_idx].stability_monitor.get_metrics()
    print(f"Confidence (Novel): {conf_B:.4f}")
    print(f"Metrics: {metrics_B}")
    
    if conf_B < conf_A:
        print("PASS: Confidence lower for novel pattern.")
    else:
        print("FAIL: Confidence not lower for novel pattern.")
        
    if conf_B < 0.5:
        print("PASS: Confidence represents uncertainty.")
    else:
        print("WARN: Confidence might be too high for novel input.")

    # 4. Verify Improvement
    print("\n--- Summary ---")
    print(f"Delta: {conf_A - conf_B:.4f}")
    if (conf_A - conf_B) > 0.2:
        print("SUCCESS: Significant discrimination between familiar and novel.")
    else:
        print("FAILURE: Poor discrimination.")

if __name__ == "__main__":
    test_confidence_accuracy()
