"""Rebuild the phaselp wheel used by the browser app and check the static site in apps/.

Run from the repository root after changing the package:  python scripts/build_site.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
version = re.search(r'^version = "(.+)"', (ROOT / "pyproject.toml").read_text(), re.M)[1]
wheel = f"phaselp-{version}-py3-none-any.whl"

for old in (APPS / "vendor").glob("phaselp-*.whl"):
    old.unlink()
subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(APPS / "vendor"), "-q"], check=True)
assert (APPS / "vendor" / wheel).exists(), wheel
assert f"vendor/{wheel}" in (APPS / "worker.js").read_text(), f"update WHEEL in apps/worker.js to vendor/{wheel}"
for name in ["index.html", "app.js", "client.js", "worker.js", "slice.js", "hover.js",
             "enhancements.css", "vendor/mrcfile-1.5.4-py2.py3-none-any.whl",
             "demo/high.mrc.gz", "demo/low.mrc.gz", "demo/manifest.json"]:
    assert (APPS / name).exists(), f"missing apps/{name}"
for page in ["index.html", "app.js", "client.js", "worker.js"]:
    text = (APPS / page).read_text()
    assert not re.search(r'''(?:src|href)=["']/''', text), f"absolute local path in {page}"
print(f"apps/vendor/{wheel} rebuilt; site files and relative paths checked.")
