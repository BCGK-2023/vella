# vella-core

The pure-data core model for the [Vella](https://github.com/vella/vella) graph SDK —
`Node`, `Edge`, references, state, tools, and the type registry. Zero
first-party dependencies (pydantic v2 + typing_extensions only), publishable
standalone so anyone can build integrations without the rest of the stack.

```bash
pip install vella-core
```

<!-- This block is an executable doctest (run in CI via `pytest --doctest-glob`). It
registers into a *local* Registry so the example stays hermetic — it never mutates
the process-global default_registry. -->

```python
>>> from uuid import uuid4
>>> from pydantic import ConfigDict
>>> from vella.core import Node, FlexibleData, node_type, ToolDeclaration, Registry
>>>
>>> registry = Registry()  # keep the example hermetic; don't touch global state
>>> @node_type(
...     "outlook_email",
...     compat="BACKWARD",
...     tools=[ToolDeclaration(name="reply_to_email", description="Reply to this email.")],
...     registry=registry,
... )
... class OutlookEmailData(FlexibleData):
...     model_config = ConfigDict(extra="forbid", frozen=True)  # strict + frozen
...     subject: str
...     body: str
>>>
>>> email = Node.from_data(
...     OutlookEmailData(subject="Q3 numbers", body="..."),
...     name="Q3 numbers",
...     created_by=uuid4(),
... )
>>> email.type
'outlook_email'
>>> email.data.subject
'Q3 numbers'
>>> email.id.version  # time-ordered UUIDv7
7

```

## Design at a glance

- **Pure, frozen data.** Nodes and edges carry no behavior; changes go through
  copy-on-write (`evolve` / `update_state` / `update_desired`) which re-validate.
- **One envelope, polymorphic via generics.** `Node[TData, TState]`. Strict
  models for system-managed types; `FlexibleData` for agent-managed ones.
- **Four doors to a node** — construct (strict), `evolve` (copy-on-write),
  `parse_node` (tolerant hydration from untyped data), `hydrate` (trusted fast
  path). `model_construct` is locked.
- **Same surface for everyone.** Internal and external integrations use this
  exact API — no privileged internal path.
- **Tolerant reader.** `parse_node` resolves types via the registry, migrates
  across schema versions, and quarantines unparseable data as a repairable
  `FlexibleData` node instead of throwing — the substrate for self-healing.
- **Structural multi-tenancy.** Every node belongs to exactly one `tenant_id`
  (default `__local__`); never null, so partition isolation is universal.
- **Time-ordered ids.** UUIDv7 for storage locality; idempotency is binding-level
  on `(tenant_id, plugin, external_id)`.

See [DESIGN.md](DESIGN.md) for the full rationale and deferred work.

## Development

```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"

# The full gate — all fail-closed, identical to CI:
pytest                                          # tests + README/docstring doctests
mypy                                            # strict
pyright                                         # strict
ruff check src/vella/core                       # Google docstring convention
interrogate -c pyproject.toml src/vella/core    # 100% public docstring coverage
mkdocs build --strict                           # API site renders clean
python scripts/export_schema.py --check         # schema breaking-change tripwire
python scripts/generate_catalogs.py --check     # generated-catalog drift tripwire
```

Generated artifacts (`schema/`, `docs/catalogs/`) are never hand-edited —
regenerate them by running the script without `--check`. The API docs site is
built locally with `mkdocs serve`; public publication is not yet enabled.

## License

Apache-2.0. See [LICENSE](LICENSE).
