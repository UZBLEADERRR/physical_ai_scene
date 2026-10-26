"""The demo web app renders every module page without an exception (Streamlit AppTest, no browser)."""

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

PAGES = ["Overview", "S1 Sensors", "S3 Perception", "S2 3D Fusion", "S6 Tracking", "S4 Scene Graph",
         "S5 Reasoning", "S6 Action", "S7 Integration"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from demo import runs

    root = tmp_path_factory.mktemp("store")
    saved = runs.STORE, runs.DATASETS, runs.RUNS
    runs.STORE, runs.DATASETS, runs.RUNS = root, root / "datasets", root / "runs"
    runs.build_builtin()
    yield runs
    runs.STORE, runs.DATASETS, runs.RUNS = saved


def test_every_page_renders(store):
    at = streamlit_testing.AppTest.from_file("../demo/app.py", default_timeout=240).run()
    assert not at.exception
    for page in PAGES:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])


def test_questions_get_verified_answers(store):
    at = streamlit_testing.AppTest.from_file("../demo/app.py", default_timeout=240).run()
    at.session_state["page"] = "S5 Reasoning"
    at.run()
    asked = 0
    for b in [b for b in at.button if b.key and b.key.startswith("q_")]:
        at = b.click().run()
        assert not at.exception, b.label
        asked += 1
    assert asked >= 3
    texts = [s.value for s in at.success] + [w.value for w in at.warning]
    assert any("Cannot determine" in t for t in texts)      # the unseen lidar scanner is refused, not guessed
