"""Settings from environment variables — the IB_* three only.

OPENAI_API_KEY and OPENAI_DEFAULT_MODEL are consumed by the Agents SDK directly and
never pass through here.
"""

import os
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class Settings:
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            ib_host=os.environ.get("IB_HOST", "127.0.0.1"),
            ib_port=int(os.environ.get("IB_PORT", "7497")),
            ib_client_id=int(os.environ.get("IB_CLIENT_ID", "1")),
        )
