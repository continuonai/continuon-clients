# Project Workflow: Continuon AI

## Task Management
- Every change must be tracked in a `plan.md` file within a specific track folder.
- Tasks should be atomic and focused on a single logical change.

## Development Loop
1. **Understand:** Analyze requirements and existing code.
2. **Plan:** Update the `plan.md` with specific tasks.
3. **Implement:** Write code, following the defined style guides.
4. **Verify:**
    - Minimum **80% code coverage** for new features.
    - Run static analysis and linting (e.g., `flutter analyze`, `pytest`).
5. **Checkpoint:**
    - Commit changes after every task.
    - Use **Git Notes** to record a brief summary of what was accomplished in the task.

## Phase Completion
- Upon completing a phase (group of tasks), perform a manual verification of the system's state before proceeding.
- Tag the repository at significant milestones.
