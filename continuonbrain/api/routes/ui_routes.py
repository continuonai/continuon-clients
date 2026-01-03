import importlib.util
from pathlib import Path
from typing import Any, Dict

import jinja2

_flask_spec = importlib.util.find_spec("flask")
if _flask_spec:
    from flask import Blueprint  # type: ignore
else:
    class Blueprint:  # type: ignore[too-many-instance-attributes]
        def __init__(self, *args, **kwargs) -> None:
            self.name = kwargs.get("name")

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

# Initialize Blueprint
ui_bp = Blueprint('ui', __name__)

# Template Directory
# Templates are in continuonbrain/server/templates/
# ui_routes.py is in continuonbrain/api/routes/
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "server" / "templates"

# Initialize Jinja2 Environment
template_loader = jinja2.FileSystemLoader(searchpath=str(TEMPLATE_DIR))
template_env = jinja2.Environment(loader=template_loader)

def render_template(template_name: str, **context) -> str:
    """Render a Jinja2 template."""
    try:
        template = template_env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        return f"<html><body><h1>Error rendering {template_name}: {e}</h1></body></html>"

# --- Routes ---

@ui_bp.route('/')
def home():
    return get_home_html()

@ui_bp.route('/safety')
def safety():
    return get_safety_html()

@ui_bp.route('/tasks')
def tasks():
    return get_tasks_html()

@ui_bp.route('/skills')
def skills():
    return get_skills_html()

@ui_bp.route('/research')
def research():
    return get_research_html()

@ui_bp.route('/api_explorer')
def api_explorer():
    return get_api_explorer_html()

# --- Helper Functions for server.py ---
# These are called directly by server.py's BaseHTTPRequestHandler

def get_home_html() -> str:
    """Legacy route - now serves v2 dashboard."""
    return get_v2_dashboard_html()

def get_safety_html() -> str:
    """Legacy route - now serves v2 safety center."""
    return get_v2_safety_html()

def get_tasks_html() -> str:
    return render_template("tasks.html", active_page="tasks")

def get_skills_html() -> str:
    return render_template("skills.html", active_page="skills")

def get_research_html() -> str:
    return render_template("research.html", active_page="research")

def get_api_explorer_html() -> str:
    return render_template("api_explorer.html", active_page="api_explorer")

def get_training_html() -> str:
    """Legacy route - now serves v2 training hub."""
    return get_v2_training_html()

def get_training_proof_html() -> str:
    return render_template("training_proof.html", active_page="training_proof")

# Legacy/Placeholder routes - all now use v2
def get_status_html() -> str:
    """Legacy route - now serves v2 dashboard."""
    return get_v2_dashboard_html()

def get_dashboard_html() -> str:
    """Legacy route - now serves v2 dashboard."""
    return get_v2_dashboard_html()

def get_chat_html() -> str:
    """Legacy route - chat is now in agent rail."""
    return get_v2_dashboard_html() 

def get_settings_html() -> str:
    """Settings page - uses existing template."""
    return render_template("settings.html", active_page="settings")

def get_manual_html() -> str:
    """Legacy route - now serves v2 control center."""
    return get_v2_control_html()

def get_brain_map_html() -> str:
    return render_template("wiring.html", active_page="wiring") # Assuming wiring.html exists

# HOPE monitoring pages
# Keep legacy /ui/hope/* URLs but serve the unified UI template so the UX is consistent.
def get_hope_performance_html() -> str:
    return render_template("hope.html", active_page="hope", hope_section="performance")

def get_hope_dynamics_html() -> str:
    return render_template("hope.html", active_page="hope", hope_section="dynamics")

def get_hope_memory_html() -> str:
    return render_template("hope.html", active_page="hope", hope_section="memory")

def get_hope_stability_html() -> str:
    return render_template("hope.html", active_page="hope", hope_section="stability")

def get_hope_training_html() -> str:
    return render_template("hope.html", active_page="hope", hope_section="training")

# ============================================
# V2 UI - Command Center Style
# ============================================

def get_v2_dashboard_html() -> str:
    """New dashboard with live metrics and RCAN status."""
    return render_template("dashboard_v2.html", active_page="dashboard")

def get_v2_control_html() -> str:
    """Control center with camera feed and manual controls."""
    return render_template("control_v2.html", active_page="control")

def get_v2_training_html() -> str:
    """Training hub with pipeline viz and teacher mode."""
    return render_template("training_v2.html", active_page="training")

def get_v2_safety_html() -> str:
    """Safety center with Ring 0 status and work authorization."""
    return render_template("safety_v2.html", active_page="safety")

def get_v2_network_html() -> str:
    """Network & RCAN page with WiFi/BT managers."""
    return render_template("network_v2.html", active_page="network")

def get_v2_agent_html() -> str:
    """Agent intelligence page with chat and knowledge map."""
    # Falls back to dashboard since agent rail is always visible
    return render_template("dashboard_v2.html", active_page="agent")

def get_v2_settings_html() -> str:
    """Settings page in v2 style."""
    # Reuse existing settings for now, can be upgraded later
    return render_template("settings.html", active_page="settings")
