import os
import socket

import pytest

from ragqa.data import make_document

# These tests use local fixtures only. Disable optional libraries' background telemetry.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex

    def allowed(address):
        return not isinstance(address, tuple) or address[0] in ("127.0.0.1", "::1", "localhost")

    def connect(sock, address):
        if not allowed(address):
            raise RuntimeError("External network disabled in tests")
        return original(sock, address)

    def connect_ex(sock, address):
        if not allowed(address):
            raise RuntimeError("External network disabled in tests")
        return original_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture
def documents():
    return [
        make_document("演示甲编号", "编号 ALPHA-01。注意事项：保留限制。", source="synthetic:a"),
        make_document("演示乙编号", "编号 BETA-02。", source="synthetic:b"),
        make_document("演示丙编号", "编号 GAMMA-03。", source="synthetic:c"),
    ]
