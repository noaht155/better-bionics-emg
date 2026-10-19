"""The Train tab's progress bar for network jobs, read from the lines the training scripts print."""
import ml  # noqa: F401
from tools import job_progress


def run(lines):
    progress = {"share": None}
    for line in lines:
        progress = job_progress(line, progress)
    return progress


def test_epochs_of_a_single_training():
    p = run(["training on 9 sessions", "epoch 3/15, loss 0.512, about 2:10 left"])
    assert p["share"] == 3 / 15 and p["left"] == "2:10"


def test_folds_count_towards_the_whole_job():
    p = run(["  fold 2/4: training without 2026-10-01_1", "    epoch 5/10, loss 0.4, about 0:30 left"])
    assert p["share"] == (1 + 5 / 10) / 4
    # A new fold starts its epochs again
    p = job_progress("  fold 3/4: training without 2026-10-02_1", p)
    assert p["share"] == 2 / 4 and p["epoch"] is None


def test_nothing_to_count_before_the_first_epoch():
    assert run(["loading 2026-10-01_150735", "120000 windows"])["share"] is None
