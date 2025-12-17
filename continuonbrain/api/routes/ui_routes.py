from typing import Dict, Any
from pathlib import Path
import jinja2
from flask import Blueprint

# Initialize Blueprint
ui_bp = Blueprint('ui', __name__)
print("DEBUG: Loading ui_routes.py with Jinja2 templates")

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
    print(f"DEBUG: get_home_html called. TEMPLATE_DIR={TEMPLATE_DIR}")
    content = render_template("home.html", active_page="home")
    print(f"DEBUG: Rendered content length: {len(content)}")
    print(f"DEBUG: Content snippet: {content[:100]}")
    return content

def get_safety_html() -> str:
    return render_template("safety.html", active_page="safety")

def get_tasks_html() -> str:
    return render_template("tasks.html", active_page="tasks")

def get_skills_html() -> str:
    return render_template("skills.html", active_page="skills")

def get_research_html() -> str:
    return render_template("research.html", active_page="research")

def get_api_explorer_html() -> str:
    return render_template("api_explorer.html", active_page="api_explorer")

def get_training_html() -> str:
    return render_template("training.html", active_page="training")

def get_training_proof_html() -> str:
    return render_template("training_proof.html", active_page="training_proof")

# Legacy/Placeholder routes (mapped to new UI or kept as is)
def get_status_html() -> str:
    return get_home_html()

def get_dashboard_html() -> str:
    return get_home_html()

def get_chat_html() -> str:
    # Chat is now an overlay/sidebar in base.html, but if a direct link is needed:
    return get_home_html() 

def get_settings_html() -> str:
    return render_template("settings.html", active_page="settings")

def get_manual_html() -> str:
    return render_template("control.html", active_page="manual") # Assuming control.html exists

def get_brain_map_html() -> str:
    return render_template("wiring.html", active_page="wiring") # Assuming wiring.html exists

# HOPE specific pages (keep the placeholders/templates we fixed earlier)
from .ui_templates import PERFORMANCE_HTML, DYNAMICS_HTML, MEMORY_HTML, STABILITY_HTML

def get_hope_performance_html() -> str:
    return PERFORMANCE_HTML

def get_hope_dynamics_html() -> str:
    return DYNAMICS_HTML

def get_hope_memory_html() -> str:
    return MEMORY_HTML

def get_hope_stability_html() -> str:
    return STABILITY_HTML

def get_hope_training_html() -> str:
    # Try to load from training_plan_page if available, else placeholder
    try:
        from continuonbrain.api.routes.training_plan_page import get_training_plan_html
        return get_training_plan_html()
    except ImportError:
        return "<html><body><h1>HOPE Training</h1><p>Not available</p></body></html>"
