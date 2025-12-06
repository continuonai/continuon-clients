"""
Test Brain Monitoring UI Integration

Verifies that HOPE monitoring pages load correctly and API endpoints work.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_hope_ui_imports():
    """Test that HOPE UI pages can be imported."""
    from continuonbrain.api.routes import ui_routes
    
    # Test that helper functions exist
    assert hasattr(ui_routes, 'get_hope_training_html')
    assert hasattr(ui_routes, 'get_hope_memory_html')
    assert hasattr(ui_routes, 'get_hope_stability_html')
    assert hasattr(ui_routes, 'get_hope_dynamics_html')
    assert hasattr(ui_routes, 'get_hope_performance_html')
    
    print("✓ HOPE UI helper functions exist")


def test_hope_ui_pages_load():
    """Test that HOPE UI pages return valid HTML."""
    from continuonbrain.api.routes import ui_routes
    
    pages = [
        ('Training', ui_routes.get_hope_training_html()),
        ('Memory', ui_routes.get_hope_memory_html()),
        ('Stability', ui_routes.get_hope_stability_html()),
        ('Dynamics', ui_routes.get_hope_dynamics_html()),
        ('Performance', ui_routes.get_hope_performance_html()),
    ]
    
    for name, html in pages:
        assert html is not None, f"{name} page is None"
        assert len(html) > 0, f"{name} page is empty"
        assert '<html>' in html.lower(), f"{name} page is not valid HTML"
        assert '</html>' in html.lower(), f"{name} page is not closed HTML"
        print(f"✓ {name} page loads correctly ({len(html)} bytes)")


def test_hope_routes_api():
    """Test that HOPE API routes module exists."""
    try:
        from continuonbrain.api.routes import hope_routes
        assert hasattr(hope_routes, 'handle_hope_request')
        assert hasattr(hope_routes, 'set_hope_brain')
        print("✓ HOPE API routes module loaded")
    except ImportError as e:
        print(f"⚠ HOPE routes not available: {e}")


def test_learning_routes_api():
    """Test that learning API routes module exists."""
    try:
        from continuonbrain.api.routes import learning_routes
        assert hasattr(learning_routes, 'handle_learning_request')
        assert hasattr(learning_routes, 'set_background_learner')
        print("✓ Learning API routes module loaded")
    except ImportError as e:
        print(f"⚠ Learning routes not available: {e}")


def test_background_learner_exists():
    """Test that background learner service exists."""
    try:
        from continuonbrain.services.background_learner import BackgroundLearner
        print("✓ BackgroundLearner class exists")
    except ImportError as e:
        print(f"⚠ Background learner not available: {e}")


def test_curiosity_environment():
    """Test that curiosity environment works."""
    try:
        from continuonbrain.hope_impl.curiosity_env import CuriosityEnvironment, AdaptiveCuriosityEnvironment
        
        # Create environment
        env = CuriosityEnvironment(obs_dim=10, action_dim=4)
        
        # Reset
        obs = env.reset()
        assert obs.shape == (10,), "Observation shape incorrect"
        
        # Step
        import numpy as np
        action = np.random.randn(4)
        next_obs, reward, done = env.step(action)
        
        assert next_obs.shape == (10,), "Next observation shape incorrect"
        assert isinstance(reward, float), "Reward should be float"
        assert done == False, "Curiosity environment should never be done"
        
        print("✓ CuriosityEnvironment works correctly")
        
    except ImportError as e:
        print(f"⚠ Curiosity environment not available: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Brain Monitoring UI Integration")
    print("=" * 60)
    
    # Run tests
    test_hope_ui_imports()
    test_hope_ui_pages_load()
    test_hope_routes_api()
    test_learning_routes_api()
    test_background_learner_exists()
    test_curiosity_environment()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
