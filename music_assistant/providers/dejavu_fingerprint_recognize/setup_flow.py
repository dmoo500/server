"""Set up a fingerprint recognition server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.setup_flow import SetupFlowError

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession


async def run_setup(session: SetupSession) -> None:
    """Collect the server address and recognition credentials."""
    data = dict(session.context.setup_data)
    errors: dict[str, str | SetupFlowError] | None = None
    while True:
        entries = [
            ConfigEntry(
                key="server_url",
                type=ConfigEntryType.STRING,
                required=True,
                value=data.get("server_url"),
            ),
            ConfigEntry(
                key="api_key",
                type=ConfigEntryType.SECURE_STRING,
                required=True,
                value=data.get("api_key"),
            ),
            ConfigEntry(
                key="application_key",
                type=ConfigEntryType.SECURE_STRING,
                required=False,
                value=data.get("application_key"),
            ),
        ]
        data.update(await session.form(entries, step_id="user", errors=errors, last_step=True))
        try:
            await session.finish(data)
            return
        except SetupFlowError as err:
            errors = {"base": err}
