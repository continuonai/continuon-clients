"""Tests for the service container."""
import pytest

from continuonbrain.services.container import (
    ServiceContainer,
    ContainerConfig,
    get_container,
    set_container,
    reset_container,
)
from continuonbrain.services.interfaces import IChatService


class TestContainerConfig:
    def test_default_config(self):
        config = ContainerConfig()
        assert config.config_dir == "/opt/continuonos/brain"
        assert config.prefer_real_hardware is True
        assert config.allow_mock_fallback is True
        assert config.lazy_load is True

    def test_custom_config(self):
        config = ContainerConfig(
            config_dir="/tmp/test",
            prefer_real_hardware=False,
            allow_mock_fallback=False,
        )
        assert config.config_dir == "/tmp/test"
        assert config.prefer_real_hardware is False
        assert config.allow_mock_fallback is False


class TestServiceContainer:
    @pytest.fixture
    def container(self):
        config = ContainerConfig(
            config_dir="/tmp/test",
            allow_mock_fallback=True,
        )
        return ServiceContainer(config)

    def test_has_default_factories(self, container):
        assert container.has("chat")
        assert container.has("hardware")
        assert container.has("perception")
        assert container.has("learning")
        assert container.has("reasoning")

    def test_not_instantiated_initially(self, container):
        assert not container.is_instantiated("chat")
        assert not container.is_instantiated("hardware")

    def test_get_creates_service(self, container):
        chat = container.get("chat")
        assert chat is not None
        assert container.is_instantiated("chat")

    def test_get_returns_same_instance(self, container):
        chat1 = container.get("chat")
        chat2 = container.get("chat")
        assert chat1 is chat2

    def test_property_access(self, container):
        chat = container.chat
        assert chat is not None
        assert isinstance(chat, IChatService)

    def test_register_custom_factory(self, container):
        class CustomChat:
            def is_available(self):
                return True

        container.register("chat", lambda: CustomChat())
        chat = container.get("chat")
        assert isinstance(chat, CustomChat)

    def test_register_instance(self, container):
        class MockChat:
            pass

        mock = MockChat()
        container.register_instance("chat", mock)
        assert container.get("chat") is mock

    def test_get_unknown_raises(self, container):
        with pytest.raises(KeyError, match="unknown"):
            container.get("unknown")

    def test_shutdown_clears_instances(self, container):
        _ = container.chat
        assert container.is_instantiated("chat")

        container.shutdown()
        assert not container.is_instantiated("chat")

    def test_get_status(self, container):
        _ = container.chat

        status = container.get_status()
        assert "services" in status
        assert "chat" in status["services"]
        assert status["services"]["chat"]["instantiated"] is True


class TestGlobalContainer:
    def teardown_method(self):
        reset_container()

    def test_get_container_creates_default(self):
        container = get_container()
        assert container is not None

    def test_get_container_returns_same(self):
        c1 = get_container()
        c2 = get_container()
        assert c1 is c2

    def test_set_container_replaces(self):
        original = get_container()

        new_container = ServiceContainer(ContainerConfig(config_dir="/new"))
        set_container(new_container)

        assert get_container() is new_container
        assert get_container() is not original

    def test_reset_container(self):
        c1 = get_container()
        reset_container()
        c2 = get_container()

        assert c1 is not c2
