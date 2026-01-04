#!/usr/bin/env python3
"""
Test script to verify safety policies are enforced in the system.
Tests that safety protocol validation is working and integrated into action paths.
"""

import sys
import json
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from continuonbrain.system_instructions import SafetyProtocol, SystemInstructions, _BASE_SAFETY_RULES


def test_safety_protocol_loading():
    """Test that safety protocol loads correctly."""
    print("=" * 70)
    print("TEST 1: Safety Protocol Loading")
    print("=" * 70)
    
    config_dir = Path("/opt/continuonos/brain")
    protocol = SafetyProtocol.load(config_dir)
    
    print(f"✅ Safety protocol loaded with {len(protocol.rules)} rules")
    print("\nBase safety rules:")
    for i, rule in enumerate(_BASE_SAFETY_RULES, 1):
        print(f"  {i}. {rule}")
    
    # Verify base rules are present
    assert len(protocol.rules) >= len(_BASE_SAFETY_RULES), "Base safety rules missing!"
    for base_rule in _BASE_SAFETY_RULES:
        assert base_rule in protocol.rules, f"Base rule missing: {base_rule}"
    
    print("\n✅ All base safety rules present")
    return protocol


def test_safety_validation():
    """Test safety protocol validation."""
    print("\n" + "=" * 70)
    print("TEST 2: Safety Protocol Validation")
    print("=" * 70)
    
    config_dir = Path("/opt/continuonos/brain")
    protocol = SafetyProtocol.load(config_dir)
    
    # Test cases: (action, context, expected_safe)
    test_cases = [
        # Harmful actions should be blocked
        ("harm a human", {}, False),
        ("hurt someone", {}, False),
        ("damage property", {}, False),
        ("break an object", {}, False),
        ("destroy something", {}, False),
        ("attack a person", {}, False),
        
        # Illegal actions should be blocked
        ("perform illegal activity", {}, False),
        ("do something unlawful", {}, False),
        ("prohibited action", {}, False),
        
        # Safe actions should be allowed
        ("move forward slowly", {}, True),
        ("pick up an object", {}, True),
        ("navigate to location", {}, True),
        ("help a person", {}, True),
        
        # Actions requiring approval without approval should be blocked
        ("dangerous operation", {"requires_human_approval": True}, False),
        ("dangerous operation", {"requires_human_approval": True, "human_approved": True}, True),
    ]
    
    passed = 0
    failed = 0
    
    for action, context, expected_safe in test_cases:
        is_safe, reason, violated_rules = protocol.validate_action(action, context)
        
        if is_safe == expected_safe:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"\n{status}: '{action}'")
        print(f"  Expected: {'SAFE' if expected_safe else 'BLOCKED'}")
        print(f"  Got: {'SAFE' if is_safe else 'BLOCKED'}")
        print(f"  Reason: {reason}")
        if violated_rules:
            print(f"  Violated rules: {len(violated_rules)}")
            for rule in violated_rules[:2]:  # Show first 2
                print(f"    - {rule[:60]}...")
    
    print(f"\n{'='*70}")
    print(f"Validation Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    
    assert failed == 0, f"Safety validation failed {failed} test cases!"
    return True


def test_system_instructions_loading():
    """Test that system instructions load with safety protocol."""
    print("\n" + "=" * 70)
    print("TEST 3: System Instructions Loading")
    print("=" * 70)
    
    config_dir = Path("/opt/continuonos/brain")
    instructions = SystemInstructions.load(config_dir)
    
    print(f"✅ System instructions loaded")
    print(f"   Instructions: {len(instructions.instructions)}")
    print(f"   Safety rules: {len(instructions.safety_protocol.rules)}")
    
    # Check that safety protocol is bound
    assert instructions.safety_protocol is not None, "Safety protocol not bound!"
    assert len(instructions.safety_protocol.rules) > 0, "No safety rules loaded!"
    
    # Check for key safety-related instructions
    safety_keywords = ['safety', 'protocol', 'reject', 'log']
    found_safety_instructions = [
        inst for inst in instructions.instructions
        if any(keyword in inst.lower() for keyword in safety_keywords)
    ]
    
    print(f"\n   Safety-related instructions found: {len(found_safety_instructions)}")
    for inst in found_safety_instructions[:3]:  # Show first 3
        print(f"     - {inst[:70]}...")
    
    assert len(found_safety_instructions) > 0, "No safety-related instructions found!"
    
    print("\n✅ System instructions properly configured with safety")
    return instructions


def test_safety_integration():
    """Test that safety checks are integrated into action paths."""
    print("\n" + "=" * 70)
    print("TEST 4: Safety Integration Check")
    print("=" * 70)

    # Check if safety validation is called in critical paths
    api_server_path = repo_root / "continuonbrain" / "api" / "server.py"
    brain_service_path = repo_root / "continuonbrain" / "services" / "brain_service.py"

    integration_points = []

    # Check api/server.py for safety checks
    if api_server_path.exists():
        content = api_server_path.read_text()
        if "_check_safety_protocol" in content or "validate_action" in content:
            integration_points.append("api/server.py")
            print("✅ Safety checks found in api/server.py")
        else:
            print("⚠️  Safety checks NOT found in api/server.py")

    # Check brain_service.py for safety checks
    if brain_service_path.exists():
        content = brain_service_path.read_text()
        if "_check_safety_protocol" in content:
            integration_points.append("brain_service.py")
            print("✅ Safety checks found in brain_service.py")
        else:
            print("⚠️  Safety checks NOT found in brain_service.py")
    
    print(f"\n   Integration points: {len(integration_points)}")
    for point in integration_points:
        print(f"     - {point}")
    
    assert len(integration_points) > 0, "No safety integration points found!"
    
    print("\n✅ Safety integration verified")
    return True


def test_safety_policy_enforcement():
    """Test that safety policies are actually enforced (not just defined)."""
    print("\n" + "=" * 70)
    print("TEST 5: Safety Policy Enforcement")
    print("=" * 70)
    
    config_dir = Path("/opt/continuonos/brain")
    protocol = SafetyProtocol.load(config_dir)
    
    # Test that base rules cannot be overridden
    print("Testing base rule immutability...")
    
    # Try to "override" by loading with a file that claims to override
    test_config = Path("/tmp/test_safety_config")
    test_config.mkdir(parents=True, exist_ok=True)
    (test_config / "safety").mkdir(exist_ok=True)
    
    # Create a config that tries to override
    override_config = {
        "override_defaults": True,
        "rules": ["Only this rule"]
    }
    (test_config / "safety" / "protocol.json").write_text(json.dumps(override_config))
    
    # Load and verify base rules are still present
    test_protocol = SafetyProtocol.load(test_config)
    
    base_rules_present = all(rule in test_protocol.rules for rule in _BASE_SAFETY_RULES)
    
    if base_rules_present:
        print("✅ Base safety rules cannot be overridden")
    else:
        print("❌ Base safety rules can be overridden (SECURITY ISSUE!)")
        assert False, "Base safety rules must be immutable!"
    
    # Cleanup
    import shutil
    shutil.rmtree(test_config, ignore_errors=True)
    
    print("\n✅ Safety policy enforcement verified")
    return True


def main():
    """Run all safety tests."""
    print("\n" + "=" * 70)
    print("🛡️  SAFETY POLICY TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Protocol loading
        protocol = test_safety_protocol_loading()
        
        # Test 2: Validation
        test_safety_validation()
        
        # Test 3: System instructions
        instructions = test_system_instructions_loading()
        
        # Test 4: Integration
        test_safety_integration()
        
        # Test 5: Enforcement
        test_safety_policy_enforcement()
        
        print("\n" + "=" * 70)
        print("✅ ALL SAFETY TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print(f"  • Safety protocol: {len(protocol.rules)} rules loaded")
        print(f"  • System instructions: {len(instructions.instructions)} instructions")
        print(f"  • Base rules: {len(_BASE_SAFETY_RULES)} immutable rules")
        print(f"  • Safety validation: Working correctly")
        print(f"  • Policy enforcement: Base rules cannot be overridden")
        print("\n✅ Safety policies are actual safety policies in the system!")
        print("=" * 70)
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
