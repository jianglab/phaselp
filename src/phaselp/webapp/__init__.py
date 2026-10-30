"""Python side of the phaselp browser app (see ``apps/`` for the page).

``session.dispatch_json`` handles every request from the page. It runs in
Pyodide inside a web worker, and in ordinary Python for testing.
"""
