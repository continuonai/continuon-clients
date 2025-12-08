def get_learning_dashboard_html() -> str:
    """Return the learning dashboard HTML."""
    with open('/home/craigm26/ContinuonXR/continuonbrain/api/routes/learning_dashboard.html', 'r') as f:
        return f.read()
