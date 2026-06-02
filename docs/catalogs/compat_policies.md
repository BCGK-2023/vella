# Compatibility policies

Generated from the `CompatPolicy` type in `vella.core`. Do not edit by hand; run `python scripts/generate_catalogs.py`.

Per-type schema-evolution policy, enforced by the schema tripwire (`scripts/export_schema.py --check`). Confluent semantics.

| Policy | Semantics |
| --- | --- |
| `FULL` | Both directions: a new reader reads old data and an old reader reads new data. |
| `BACKWARD` | A new reader reads old data. |
| `FORWARD` | An old reader reads new data. |
| `NONE` | No compatibility check (schema-on-read). |
| `FULL_TRANSITIVE` | Both directions: a new reader reads old data and an old reader reads new data. Checked across the whole version history (not just the adjacent version). |
| `BACKWARD_TRANSITIVE` | A new reader reads old data. Checked across the whole version history (not just the adjacent version). |
| `FORWARD_TRANSITIVE` | An old reader reads new data. Checked across the whole version history (not just the adjacent version). |
