# Contributing
State topology, current/power signs, units, port conventions, switching/sampling timestamps and observed quantities. Changes to power physics need independent analytic, conservation or solver comparisons. Preserve adverse cases and declared approximations.

Run `uv sync --locked --extra dev`, `uv run ruff check src tests examples`, `uv run ruff format --check src tests examples` and `uv run pytest -q`. Regenerate numerical evidence from one clean source revision, then render, verify and publish with the gallery builder. Never edit sealed summaries or provenance by hand.

Keep credentials, private hardware configuration and generated outputs outside reviewed artifacts. Hardware adapters should remain optional, test their packet contracts independently and make no unverified real-time claims.
