"""
ContinuonBrain Safety Kernel

The Safety Kernel operates at Ring 0 (highest privilege) and:
- Initializes FIRST on boot, before any other component
- Cannot be disabled or bypassed by any other system
- Has veto power over ALL actions
- Runs with highest OS priority (real-time if available)
- Monitors all sensor inputs for safety violations
- Can trigger emergency stop at any time

Architecture (Unix/Linux privilege model):
┌─────────────────────────────────────────────────────────────────┐
│ Ring 0 - SAFETY KERNEL (This module)                            │
│ - Emergency stop, safety bounds, protocol enforcement           │
│ - Cannot be killed, overridden, or bypassed                     │
├─────────────────────────────────────────────────────────────────┤
│ Ring 1 - HARDWARE ABSTRACTION                                   │
│ - Sensor drivers, actuator interfaces                           │
│ - Safety kernel has direct access                               │
├─────────────────────────────────────────────────────────────────┤
│ Ring 2 - CORE RUNTIME                                           │
│ - WaveCore, CMS Memory, Context Graph                           │
│ - All actions filtered through Ring 0                           │
├─────────────────────────────────────────────────────────────────┤
│ Ring 3 - USER SPACE                                             │
│ - Chat, UI, API, Applications                                   │
│ - Lowest privilege, cannot modify safety                        │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from continuonbrain.safety import SafetyKernel
    
    # Safety kernel initializes automatically on import
    # It registers as the first boot component
    
    # All actions must pass through safety gate
    if SafetyKernel.allow_action(action):
        execute(action)
    else:
        handle_safety_violation(action)
    
    # Emergency stop (always works, cannot be blocked)
    SafetyKernel.emergency_stop()
"""

from .kernel import SafetyKernel, SafetyViolation, EmergencyStop
from .protocol import SafetyProtocol, PROTOCOL_66, ProtocolCategory
from .bounds import SafetyBounds, WorkspaceBounds
from .monitor import SafetyMonitor
from .work_authorization import (
    WorkAuthorizationManager,
    DestructiveActionType,
    DestructiveActionGuard,
    PropertyClaim,
)
from .anti_subversion import (
    AntiSubversionGuard,
    ImmutableSafetyCore,
    TamperEvidentLog,
    MultiPartyAuthorizer,
    SecurityAnomalyDetector,
    PromptInjectionDefense,
    SecurityError,
)

# Initialize safety kernel on import (Ring 0 - first component)
_kernel = SafetyKernel._get_instance()

__all__ = [
    # Core safety
    'SafetyKernel',
    'SafetyViolation', 
    'EmergencyStop',
    'SafetyProtocol',
    'PROTOCOL_66',
    'ProtocolCategory',
    'SafetyBounds',
    'WorkspaceBounds',
    'SafetyMonitor',
    # Work authorization
    'WorkAuthorizationManager',
    'DestructiveActionType',
    'DestructiveActionGuard',
    'PropertyClaim',
    # Anti-subversion
    'AntiSubversionGuard',
    'ImmutableSafetyCore',
    'TamperEvidentLog',
    'MultiPartyAuthorizer',
    'SecurityAnomalyDetector',
    'PromptInjectionDefense',
    'SecurityError',
]

