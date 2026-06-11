"""Adapter glue tests. Resolver logic is covered by the vendored resolver tests;
these cover arg mapping, plan-outcome handling, and client execution."""
import pytest

from moneypenny.homey.aliases import HomeyAliases
from moneypenny.homey.catalog import CapabilityRecord, DeviceRecord, HomeyCatalog
from moneypenny.homey.client import HomeyClientError
from moneypenny.homey.resolver import HomeyResolver
from moneypenny.tools.homey_adapter import HomeyAdapter, HomeyResult


class FakeClient:
    """Records writes; stands in for the live Homey box (third-party fake)."""

    def __init__(self):
        self.calls = []

    def set_capability_value(self, device_id, capability_id, value):
        self.calls.append((device_id, capability_id, value))
        return {"value": value}


class ExplodingClient(FakeClient):
    """Fake whose writes always fail at the HTTP boundary."""

    def set_capability_value(self, device_id, capability_id, value):
        raise HomeyClientError("boom")


def _cap(cid, ctype, value=None, **kw):
    return CapabilityRecord(
        id=cid, title=cid, type=ctype, value=value,
        min=kw.get("min"), max=kw.get("max"),
        enum_value_ids=kw.get("enum_value_ids", ()),
        writable=True, setable=True, readable=True, getable=True,
    )


def _resolver():
    lamp = DeviceRecord(
        id="d1", name="Desk Lamp", zone_id="z1", zone_name="Office",
        class_name="light",
        capabilities=(
            _cap("onoff", "boolean", value=False),
            _cap("dim", "number", value=0.5, min=0, max=1),
        ),
    )
    return HomeyResolver(HomeyCatalog((lamp,)), HomeyAliases.empty())


@pytest.fixture
def adapter():
    client = FakeClient()
    return HomeyAdapter(client, _resolver()), client


def test_turn_on_device(adapter):
    a, client = adapter
    result = a.execute(action="turn_on", device="desk lamp")
    assert isinstance(result, HomeyResult)
    assert result.ok
    assert ("d1", "onoff", True) in client.calls
    assert "DESK LAMP" in result.summary.upper()


def test_dim_via_capability_path(adapter):
    a, client = adapter
    result = a.execute(action="set", device="desk lamp", capability="dim", value=0.3)
    assert result.ok
    assert ("d1", "dim", 0.3) in client.calls


def test_unresolvable_returns_failure_not_exception(adapter):
    a, client = adapter
    result = a.execute(action="turn_on", device="flux capacitor")
    assert not result.ok
    assert client.calls == []
    assert result.summary  # something a briefing can carry


def test_unsupported_action_fails_cleanly(adapter):
    a, client = adapter
    result = a.execute(action="defenestrate", device="desk lamp")
    assert not result.ok
    assert client.calls == []


def test_broad_zone_plan_requires_confirmation_and_is_refused(adapter):
    # "all" + zone is a broad target: the resolver marks the plan
    # requires_confirmation, which Tier 1 refuses without touching the client.
    a, client = adapter
    result = a.execute(action="turn_on", device="all", zone="office")
    assert not result.ok
    assert "NEEDS CONFIRMATION" in result.summary
    assert client.calls == []


def test_client_error_yields_failure_result_not_exception():
    client = ExplodingClient()
    a = HomeyAdapter(client, _resolver())
    result = a.execute(action="turn_on", device="desk lamp")
    assert not result.ok
    assert "DESK LAMP" in result.summary.upper()


def test_status_query_is_refused_as_unsupported(adapter):
    a, client = adapter
    result = a.execute(action="status", device="desk lamp")
    assert not result.ok
    assert "STATUS QUERY NOT SUPPORTED" in result.summary
    assert client.calls == []
