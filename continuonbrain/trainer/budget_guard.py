"""
Lightweight budget guard to cap downloads/API calls during on-device runs.

Defaults enforce:
- max_download_bytes: 500 MB
- max_api_calls: 5 (e.g., Gemini/tool calls)

Callers should invoke `consume_download(bytes_)` and `consume_api_call()` to
track usage; `check_limits()` raises if a budget is exceeded.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class BudgetGuard:
    max_download_bytes: int = 500 * 1024 * 1024  # 500 MB
    max_api_calls: int = 5
    download_used: int = 0
    api_calls: int = 0

    def consume_download(self, bytes_: int) -> None:
        self.download_used += max(0, int(bytes_))
        self.check_limits()

    def consume_api_call(self) -> None:
        self.api_calls += 1
        self.check_limits()

    def check_limits(self) -> None:
        if self.download_used > self.max_download_bytes:
            raise RuntimeError(
                f"Download budget exceeded ({self.download_used} > {self.max_download_bytes} bytes)"
            )
        if self.api_calls > self.max_api_calls:
            raise RuntimeError(
                f"API call budget exceeded ({self.api_calls} > {self.max_api_calls})"
            )

    def snapshot(self) -> dict:
        return asdict(self)


