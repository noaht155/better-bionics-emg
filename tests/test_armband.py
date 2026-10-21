"""The armband reader process (armband.Armband), on MindRove's synthetic board so no hardware is needed."""
import time

import pytest

import ml  # noqa: F401
from armband import Armband


def wait_for_samples(band, seconds=3.0):
    got, end = 0, time.time() + seconds
    while time.time() < end and not got:
        got = band.read_all().shape[1]
        time.sleep(0.05)
    return got


@pytest.fixture
def band():
    b = Armband(synthetic=True)
    yield b
    b.close()


def test_reader_streams_all_rows(band):
    band.open()
    assert band.connected
    time.sleep(0.5)
    data = band.read_all()
    assert data.shape[0] == band.rows["num_rows"] and data.shape[1] > 0
    assert band.read().shape[0] == len(band.channels)


def test_close_and_reopen(band):
    band.open()
    assert wait_for_samples(band)
    band.close()
    assert not band.connected
    band.read_all()
    time.sleep(0.3)
    assert band.read_all().shape[1] == 0
    band.open()
    assert band.connected and wait_for_samples(band)


def test_reader_dying_leaves_the_app_running(band):
    """A library crash only ends the reader: the band reads as not connected and returns no data."""
    band.open()
    band._proc.kill()
    band._proc.wait()
    assert not band.connected
    time.sleep(0.2)
    band.read_all()
    assert band.read_all().shape[1] == 0
    band.open()
    assert wait_for_samples(band)


def test_context_manager_closes_the_reader():
    with Armband(synthetic=True) as b:
        proc = b._proc
        assert b.connected
    assert proc.poll() is not None


def test_mode_switch_keeps_streaming(band):
    # The synthetic board refuses mode commands; the reader has to carry on streaming
    band.open()
    band.set_mode("TEST")
    band.set_mode("EEG")
    band.read_all()
    assert wait_for_samples(band) > 0 and band.connected
