"""Throwaway dead-code reachability probe for src/reports/dashboard.py."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent
DASH = ROOT / "src" / "reports" / "dashboard.py"

source = DASH.read_text(encoding="utf-8")
tree = ast.parse(source)

# Top-level functions: name -> (start, end)
funcs: dict[str, tuple[int, int]] = {}
func_nodes: dict[str, ast.AST] = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        funcs[node.name] = (node.lineno, node.end_lineno)
        func_nodes[node.name] = node

names = set(funcs)

# Edges: for each top-level func, which other top-level funcs does it reference by bare name?
edges: dict[str, set[str]] = {}
for fname, fnode in func_nodes.items():
    refs: set[str] = set()
    for sub in ast.walk(fnode):
        if isinstance(sub, ast.Name) and sub.id in names and sub.id != fname:
            refs.add(sub.id)
    edges[fname] = refs

# Module-level references outside any function (e.g. STATIC_DASHBOARD_PAGES dict, decorators)
module_refs: set[str] = set()
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in names:
            module_refs.add(sub.id)

# External roots: any dashboard function name mentioned in any other .py file in the repo.
external: set[str] = set()
for path in ROOT.rglob("*.py"):
    if path == DASH or path.name == "_deadcode_probe.py":
        continue
    if ".venv" in path.parts or "site-packages" in path.parts:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        continue
    for name in names:
        if name in text:
            external.add(name)

# Roots: public (no leading underscore) + module-level refs + external refs.
roots: set[str] = set()
for name in names:
    if not name.startswith("_"):
        roots.add(name)
roots |= module_refs
roots |= external

# Sanity: task-mandated keepers must end up LIVE.
keepers = {
    "build_dashboard_html", "write_dashboard", "write_static_dashboard_pages",
    "_build_league_tab", "_inactive_leagues_section", "_advanced_reports_section",
    "_build_recorded_games_tab", "_build_sports_market_research_dashboard_html",
}

# BFS reachability.
live: set[str] = set()
stack = list(roots)
while stack:
    cur = stack.pop()
    if cur in live:
        continue
    live.add(cur)
    for nxt in edges.get(cur, ()):  # noqa
        if nxt not in live:
            stack.append(nxt)

dead = sorted(set(names) - live, key=lambda n: funcs[n][0])

print(f"Total top-level functions: {len(names)}")
print(f"External roots ({len(external)}): {sorted(external)}")
print(f"Live functions: {len(live)}   Dead functions: {len(dead)}")
print()
print("=== KEEPER SANITY (all must be LIVE) ===")
for k in sorted(keepers):
    status = "LIVE" if k in live else "*** DEAD (PROBLEM) ***"
    print(f"  {k}: {status}")
print()
print("=== DEAD FUNCTIONS (line range, count) ===")
total_dead_lines = 0
for name in dead:
    start, end = funcs[name]
    span = end - start + 1
    total_dead_lines += span
    print(f"  {start:>5}-{end:<5} ({span:>4} lines)  {name}")
print()
print(f"Total dead lines: {total_dead_lines}  (~{total_dead_lines/len(source.splitlines())*100:.0f}% of file)")
