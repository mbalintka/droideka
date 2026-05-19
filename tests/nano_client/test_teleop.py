import math
import time
import pytest
from nano_client.controller.teleop import TeleopState, _DEADMAN_MS

TARGET_V = 0.3
MAX_STEER = math.radians(30.0)


@pytest.fixture
def state() -> TeleopState:
    return TeleopState(target_v=TARGET_V, max_steer_rad=MAX_STEER)


def test_initial_twist_is_zero(state):
    v, delta = state.compute_twist()
    assert v == 0.0
    assert delta == 0.0


def test_no_deadman_fires_immediately_after_init(state):
    # Forcibly set a non-zero speed; compute_twist at 'now' must not zero it
    # because _last_key_ns was initialised to time.time_ns() at construction.
    state._cmd_v = TARGET_V
    v, _ = state.compute_twist(now_ns=time.time_ns())
    assert v == pytest.approx(TARGET_V)


def test_w_sets_forward(state):
    state.handle_key("w")
    v, delta = state.compute_twist()
    assert v == pytest.approx(TARGET_V)
    assert delta == pytest.approx(0.0)


def test_s_sets_reverse(state):
    state.handle_key("s")
    v, _ = state.compute_twist()
    assert v == pytest.approx(-TARGET_V)


def test_a_sets_left(state):
    state.handle_key("a")
    _, delta = state.compute_twist()
    assert delta == pytest.approx(-MAX_STEER)


def test_d_sets_right(state):
    state.handle_key("d")
    _, delta = state.compute_twist()
    assert delta == pytest.approx(MAX_STEER)


def test_space_zeros_both_axes(state):
    state.handle_key("w")
    state.handle_key("d")
    state.handle_key(" ")
    v, delta = state.compute_twist()
    assert v == 0.0
    assert delta == 0.0


def test_wd_combined(state):
    state.handle_key("w")
    state.handle_key("d")
    v, delta = state.compute_twist()
    assert v == pytest.approx(TARGET_V)
    assert delta == pytest.approx(MAX_STEER)


def test_deadman_fires_after_timeout(state):
    state.handle_key("w")
    # Simulate a future timestamp beyond the deadman window.
    stale_ns = time.time_ns() + int((_DEADMAN_MS + 50) * 1_000_000)
    v, delta = state.compute_twist(now_ns=stale_ns)
    assert v == 0.0
    assert delta == 0.0


def test_q_returns_true(state):
    assert state.handle_key("q") is True


def test_normal_keys_return_false(state):
    assert state.handle_key("w") is False
    assert state.handle_key("x") is False  # unknown key — no quit, no crash


def test_case_insensitive(state):
    state.handle_key("W")
    v, _ = state.compute_twist()
    assert v == pytest.approx(TARGET_V)


def test_deadman_constant_exported():
    # Smoke: the module-level constant is a positive int so tests can use it.
    assert isinstance(_DEADMAN_MS, int)
    assert _DEADMAN_MS > 0
