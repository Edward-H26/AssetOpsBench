"""A server named by its entry point is launched with the stub of uv, never with the real one (conftest.py of the repository)."""
import shutil
from pathlib import Path

import pytest

from agent.plan_execute.executor import _REPO_ROOT, _list_tools


def test_uv_on_the_path_of_the_tests_is_the_stub(uv_stub):
    assert Path(shutil.which("uv")).parent == uv_stub.parent


@pytest.mark.anyio
async def test_a_server_named_by_its_entry_point_reaches_the_stub_and_not_the_real_uv(uv_stub):
    with pytest.raises(Exception):
        await _list_tools("iot-mcp-server")
    assert f"{_REPO_ROOT} run iot-mcp-server" in uv_stub.read_text().splitlines()
