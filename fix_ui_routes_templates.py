from typing import Dict, Any
from pathlib import Path
import jinja2
from flask import Blueprint

# Initialize Blueprint
ui_bp = Blueprint('ui', __name__)

# Template Directory
# Assuming this file is in continuonbrain/api/routes/
# Templates are in continuonbrain/server/templates/
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "server" / "templates"

def get_template_content(filename: str) -> str:
    """Read a template file from disk and return its content."""
    try:
        file_path = TEMPLATE_DIR / filename
        if not file_path.exists():
            return f"<html><body><h1>Error: {filename} not found</h1><p>Path: {file_path}</p></body></html>"
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error reading {filename}: {e}</h1></body></html>"

# --- Routes ---

@ui_bp.route('/')
def home():
    return get_home_html()

@ui_bp.route('/performance')
def performance():
    return get_hope_performance_html()

@ui_bp.route('/dynamics')
def dynamics():
    return get_hope_dynamics_html()

@ui_bp.route('/memory')
def memory():
    return get_hope_memory_html()

@ui_bp.route('/stability')
def stability():
    return get_hope_stability_html()

@ui_bp.route('/training')
def training():
    return get_hope_training_html()

# --- Helper Functions for server.py ---
# These are called directly by server.py's BaseHTTPRequestHandler

def get_home_html() -> str:
    # Serve the main SPA UI
    return get_template_content("ui.html")

def get_status_html() -> str:
    return get_template_content("ui.html")

def get_dashboard_html() -> str:
    return get_template_content("ui.html")

def get_chat_html() -> str:
    return get_template_content("ui.html")

def get_settings_html() -> str:
    return get_template_content("ui.html")

def get_manual_html() -> str:
    return get_template_content("ui.html")

def get_tasks_html() -> str:
    return get_template_content("ui.html")

def get_brain_map_html() -> str:
    return get_template_content("ui.html")

# HOPE specific pages (keep the placeholders/templates we fixed earlier if they are useful, 
# or point to ui.html if they are integrated)
# The user's original issue was about these specific HTML strings.
# They seem to be distinct dashboards. I should keep them as they were (imported from ui_templates).

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
