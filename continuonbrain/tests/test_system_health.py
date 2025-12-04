from pathlib import Path

import types

from continuonbrain.system_health import HealthStatus, SystemHealthChecker


def test_detect_code_editors_prefers_cli(monkeypatch):
    checker = SystemHealthChecker()

    monkeypatch.setattr(
        "continuonbrain.system_health.shutil.which",
        lambda binary: "/usr/bin/code" if binary == "code" else None,
    )

    editors, detail = checker._detect_code_editors()

    assert "VS Code" in editors
    assert detail is None


def test_detect_code_editors_uses_process_hint(monkeypatch):
    checker = SystemHealthChecker()

    monkeypatch.setattr("continuonbrain.system_health.shutil.which", lambda _binary: None)

    result = types.SimpleNamespace(stdout="vscode\npython\n", returncode=0)
    monkeypatch.setattr(
        "continuonbrain.system_health.subprocess.run",
        lambda *args, **kwargs: result,
    )

    editors, detail = checker._detect_code_editors()

    assert editors == []
    assert detail == "Found running visual editor process"


def test_detect_mcp_tooling_detects_configs(monkeypatch, tmp_path: Path):
    checker = SystemHealthChecker(config_dir=str(tmp_path))

    monkeypatch.setattr("continuonbrain.system_health.shutil.which", lambda _binary: None)

    mcp_dir = tmp_path / "mcp"
    mcp_dir.mkdir()

    detection = checker._detect_mcp_tooling()

    assert detection["available"] is True
    assert str(mcp_dir) in detection["configs"]
    assert detection["binaries"] == []


def test_detect_gemini_cli_prefers_binary(monkeypatch):
    checker = SystemHealthChecker()

    monkeypatch.setattr(
        "continuonbrain.system_health.shutil.which",
        lambda binary: "/usr/bin/gemini" if binary == "gemini" else None,
    )

    detection = checker._detect_gemini_cli()

    assert detection["available"] is True
    assert "gemini" in detection["binaries"]
    assert detection["configs"] == []


def test_self_update_tooling_uses_gemini_parent(monkeypatch, tmp_path: Path):
    checker = SystemHealthChecker(config_dir=str(tmp_path))

    monkeypatch.setattr(
        "continuonbrain.system_health.shutil.which",
        lambda binary: "/usr/bin/gemini" if binary == "gemini" else None,
    )

    checker._check_self_update_tooling()

    result = checker.results[-1]
    assert result.component == "Self-Update"
    assert result.status is HealthStatus.HEALTHY
    assert result.details["gemini_cli"]["available"] is True
    assert result.details["local_brain_present"] is True


def test_mission_statement_check_loads_guardrail():
    checker = SystemHealthChecker()

    checker._check_mission_statement()

    result = checker.results[-1]
    assert result.component == "Mission Statement"
    assert result.status is HealthStatus.HEALTHY
    assert result.details["present"] is True
    assert "Continuon AI" in result.details["headline"]


def test_api_budget_defaults_to_five(monkeypatch, tmp_path: Path):
    checker = SystemHealthChecker(config_dir=str(tmp_path))

    monkeypatch.delenv("BRAIN_DAILY_API_BUDGET_USD", raising=False)

    checker._check_api_budget()

    result = checker.results[-1]
    assert result.component == "API Budget"
    assert result.status is HealthStatus.HEALTHY
    assert result.details["daily_limit_usd"] == 5.0
    assert result.details["source"] == "default"


def test_api_budget_reads_config(monkeypatch, tmp_path: Path):
    checker = SystemHealthChecker(config_dir=str(tmp_path))

    monkeypatch.delenv("BRAIN_DAILY_API_BUDGET_USD", raising=False)

    budget_dir = tmp_path / "budgets"
    budget_dir.mkdir()
    (budget_dir / "api_budget.json").write_text("{\n  \"daily_limit_usd\": 3.25\n}")

    checker._check_api_budget()

    result = checker.results[-1]
    assert result.component == "API Budget"
    assert result.details["daily_limit_usd"] == 3.25
    assert result.details["source"].startswith("config:")
    assert result.status is HealthStatus.HEALTHY


def test_api_budget_warns_when_high(monkeypatch, tmp_path: Path):
    checker = SystemHealthChecker(config_dir=str(tmp_path))

    monkeypatch.setenv("BRAIN_DAILY_API_BUDGET_USD", "12.5")

    checker._check_api_budget()

    result = checker.results[-1]
    assert result.component == "API Budget"
    assert result.status is HealthStatus.WARNING
    assert result.details["daily_limit_usd"] == 12.5
