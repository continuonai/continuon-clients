"""
Authorized Robot Builder

Integrates the robot builder with the safety work authorization system.
All robot construction requires explicit owner authorization.

This ensures:
1. Only authorized owners can build new robots
2. All parts are owned by the authorizing party
3. Construction is logged in tamper-evident audit trail
4. Multi-party approval for critical builds
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid

from continuonbrain.swarm.builder import (
    RobotBuilder,
    BuildPlan,
    PartInventory,
    Part,
    RobotArchetype,
)
from continuonbrain.swarm.replicator import SeedReplicator, CloneJob
from continuonbrain.safety.work_authorization import (
    WorkAuthorizationManager,
    DestructiveActionType,
    PropertyClaim,
    WorkAuthorization,
)
from continuonbrain.safety.anti_subversion import (
    AntiSubversionGuard,
    MultiPartyAuthorizer,
    TamperEvidentLog,
)

logger = logging.getLogger(__name__)


class AuthorizedRobotBuilder:
    """
    Robot builder with integrated safety authorization.
    
    All robot construction goes through safety checks:
    1. Owner role verification
    2. Parts ownership verification
    3. Multi-party approval for the build
    4. Cryptographic audit logging
    """
    
    def __init__(self):
        self.builder = RobotBuilder()
        self.replicator = SeedReplicator()
        self.auth_manager = WorkAuthorizationManager()
        self.multiparty = MultiPartyAuthorizer()
        self.guard = AntiSubversionGuard()
        self.audit_log = TamperEvidentLog()
        
        logger.info("AuthorizedRobotBuilder initialized")
    
    def register_parts(
        self,
        owner_id: str,
        owner_role: str,
        parts: List[Part],
    ) -> Tuple[bool, str, Optional[PartInventory]]:
        """
        Register parts inventory with ownership verification.
        
        Args:
            owner_id: ID of the owner registering parts
            owner_role: Role of the owner (must be creator/owner/leasee)
            parts: List of parts to register
        
        Returns:
            (success, message, inventory)
        """
        # Verify role
        if owner_role not in ["creator", "owner", "leasee"]:
            self.audit_log.log_event(
                event_type="parts_registration_denied",
                actor_id=owner_id,
                action="register_parts",
                target="inventory",
                result="denied",
                details={'reason': 'invalid_role', 'role': owner_role}
            )
            return False, f"Role '{owner_role}' cannot register parts inventory", None
        
        # Register inventory
        inventory = self.builder.register_inventory(owner_id, parts)
        
        # Log success
        self.audit_log.log_event(
            event_type="parts_registration",
            actor_id=owner_id,
            action="register_parts",
            target="inventory",
            result="success",
            details={'part_count': len(parts)}
        )
        
        return True, f"Registered {len(parts)} parts", inventory
    
    def request_build(
        self,
        owner_id: str,
        owner_role: str,
        archetype: RobotArchetype,
        robot_name: str,
        description: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Request authorization to build a robot.
        
        Creates a multi-party authorization request that must be approved.
        
        Returns:
            (success, message, request_id)
        """
        # Verify role
        if owner_role not in ["creator", "owner"]:
            return False, f"Role '{owner_role}' cannot authorize robot construction", None
        
        # Check parts availability
        analysis = self.builder.analyze_parts(owner_id)
        if analysis.get('error'):
            return False, analysis['error'], None
        
        if not analysis.get('can_build_robot'):
            missing = analysis.get('missing_parts', [])
            return False, f"Missing required parts: {', '.join(missing)}", None
        
        # Create multi-party authorization request
        # Robot construction is considered critical (needs 2 approvers)
        request = self.multiparty.create_request(
            action_type="robot_construction",
            target_id=f"new_{archetype.value}_{robot_name}",
            target_description=description,
            requester_id=owner_id,
            is_critical=True,  # Requires 2 confirmations
        )
        
        # Log the request
        self.audit_log.log_event(
            event_type="build_request_created",
            actor_id=owner_id,
            action="request_build",
            target=robot_name,
            result="pending",
            details={
                'archetype': archetype.value,
                'request_id': request.request_id,
                'required_confirmations': request.required_confirmations,
            }
        )
        
        return True, f"Build request created. Needs {request.required_confirmations} confirmations.", request.request_id
    
    def confirm_build(
        self,
        request_id: str,
        confirmer_id: str,
        confirmer_role: str,
        confirmation_method: str = "verbal",
    ) -> Tuple[bool, str]:
        """
        Confirm a build request.
        
        Must be a different person from the requester.
        """
        if confirmer_role not in ["creator", "owner"]:
            return False, f"Role '{confirmer_role}' cannot confirm robot construction"
        
        success, message = self.multiparty.add_confirmation(
            request_id=request_id,
            confirmer_id=confirmer_id,
            confirmer_role=confirmer_role,
            confirmation_method=confirmation_method,
        )
        
        self.audit_log.log_event(
            event_type="build_confirmation",
            actor_id=confirmer_id,
            action="confirm_build",
            target=request_id,
            result="success" if success else "failed",
            details={'message': message}
        )
        
        return success, message
    
    def start_build(
        self,
        request_id: str,
        owner_id: str,
        owner_role: str,
        archetype: RobotArchetype,
        robot_name: str,
    ) -> Tuple[bool, str, Optional[BuildPlan]]:
        """
        Start an approved build.
        
        Requires the build request to be fully approved.
        """
        # Check approval
        approved, reason = self.multiparty.is_approved(request_id)
        if not approved:
            return False, f"Build not approved: {reason}", None
        
        # Generate build plan
        plan = self.builder.generate_build_plan(
            owner_id=owner_id,
            archetype=archetype,
            name=robot_name,
        )
        
        if not plan:
            return False, "Failed to generate build plan", None
        
        # Auto-approve since we have multi-party approval
        self.builder.approve_plan(plan.plan_id, owner_id, owner_role)
        
        # Create work authorization for any destructive steps
        # (e.g., cutting materials, modifying parts)
        claims = [
            PropertyClaim(
                property_id=part_id,
                property_type="robot_part",
                owner_id=owner_id,
                owner_role=owner_role,
                evidence_type="inventory_registration",
            )
            for part_id in plan.required_parts
        ]
        
        work_auth = self.auth_manager.create_authorization(
            work_order_id=f"BUILD-{plan.plan_id[:8]}",
            authorizer_id=owner_id,
            authorizer_role=owner_role,
            action_types=[
                DestructiveActionType.CUT,
                DestructiveActionType.MODIFY,
                DestructiveActionType.DISASSEMBLE,
            ],
            target_properties=plan.required_parts,
            target_description=f"Construction of {robot_name}",
            property_claims=claims,
            valid_hours=float(plan.estimated_total_time_minutes) / 60 + 2,  # Buffer time
        )
        
        if work_auth:
            self.auth_manager.activate_authorization(
                work_auth.authorization_id,
                owner_id,
                owner_role,
            )
        
        self.audit_log.log_event(
            event_type="build_started",
            actor_id=owner_id,
            action="start_build",
            target=robot_name,
            result="success",
            details={
                'plan_id': plan.plan_id,
                'steps': len(plan.steps),
                'estimated_minutes': plan.estimated_total_time_minutes,
            }
        )
        
        return True, f"Build started. Plan ID: {plan.plan_id}", plan
    
    def clone_for_new_robot(
        self,
        owner_id: str,
        owner_role: str,
        parent_robot_id: str,
        target_device: str,
    ) -> Tuple[bool, str, Optional[CloneJob]]:
        """
        Clone seed image for a new robot.
        
        Requires owner authorization.
        """
        if owner_role not in ["creator", "owner"]:
            return False, f"Role '{owner_role}' cannot authorize seed cloning", None
        
        # Create clone job
        job = self.replicator.create_clone_job(
            target_device=target_device,
            parent_robot_id=parent_robot_id,
            owner_id=owner_id,
        )
        
        if not job:
            return False, "Failed to create clone job - check target device", None
        
        # Create work authorization for data writing
        claim = PropertyClaim(
            property_id=target_device,
            property_type="storage_device",
            owner_id=owner_id,
            owner_role=owner_role,
            evidence_type="physical_possession",
        )
        
        work_auth = self.auth_manager.create_authorization(
            work_order_id=f"CLONE-{job.job_id[:8]}",
            authorizer_id=owner_id,
            authorizer_role=owner_role,
            action_types=[
                DestructiveActionType.OVERWRITE,
            ],
            target_properties=[target_device],
            target_description=f"Clone seed image to new storage",
            property_claims=[claim],
            valid_hours=2.0,
        )
        
        if work_auth:
            self.auth_manager.activate_authorization(
                work_auth.authorization_id,
                owner_id,
                owner_role,
            )
        
        # Start the clone
        success, message = self.replicator.start_clone_job(job.job_id)
        
        self.audit_log.log_event(
            event_type="seed_clone",
            actor_id=owner_id,
            action="clone_seed",
            target=target_device,
            result="success" if success else "failed",
            details={
                'job_id': job.job_id,
                'new_robot_id': job.target_robot_id,
                'parent_robot_id': parent_robot_id,
            }
        )
        
        if success:
            return True, f"Clone complete. New robot ID: {job.target_robot_id}", job
        else:
            return False, message, job
    
    def get_build_capabilities(self, owner_id: str) -> Dict[str, Any]:
        """
        Get what the owner can build with available parts.
        """
        return self.builder.analyze_parts(owner_id)
    
    def list_available_devices(self) -> List[Dict]:
        """
        List storage devices available for cloning.
        """
        devices = self.replicator.detect_storage_devices()
        return [
            {
                'path': d.device_path,
                'name': d.device_name,
                'capacity_gb': d.capacity_gb,
                'removable': d.is_removable,
                'has_os': d.has_existing_os,
            }
            for d in devices
        ]


# Convenience function
def create_authorized_builder() -> AuthorizedRobotBuilder:
    """Create an authorized robot builder instance."""
    return AuthorizedRobotBuilder()

