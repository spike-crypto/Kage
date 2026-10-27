"""Give the test run its own database, before anything imports the app.

The tests talk to the real app, and until now that meant the real database. It cost us
three separate incidents: a blanket ``routing.reset()`` wiped provider measurements the
user had actually made, a routing label missing from the fixture's allow-list was never
cleaned up and stayed in the live table as a permanent fake endpoint, and three tests
gave false passes because stale dev services held the default ports. Scoping every
write to a hand-maintained list of labels is a discipline that has to be remembered
every time; a separate file has to be remembered once.

``backend.config`` reads ``KAGE_DB`` while it is being imported, and ``backend.database``
builds its engine from the result at import time too. So this has to happen before the
first ``import backend...`` anywhere in the run, which is why it sits at module level in
conftest.py rather than in a fixture: pytest imports this file before it imports any
test module.

``setdefault`` leaves an explicit ``KAGE_DB`` alone, so pointing a run at a copy of real
data to reproduce something still works.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_TMP_DIR = tempfile.mkdtemp(prefix="kage-tests-")
os.environ.setdefault("KAGE_DB", str(Path(_TMP_DIR) / "kage.db"))

# Keep the ports off the defaults too. Three tests once passed against stale dev
# services that happened to be listening on 8000 and 8501.
os.environ.setdefault("KAGE_PORT", "8799")
os.environ.setdefault("KAGE_UI_PORT", "8798")

atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)


def pytest_report_header(config) -> str:
    """Say which database the run is using, so an isolated run is never in doubt."""
    del config
    return f"kage: database = {os.environ['KAGE_DB']}"
