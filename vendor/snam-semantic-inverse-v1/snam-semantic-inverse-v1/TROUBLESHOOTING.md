# Troubleshooting / known boundaries

- **No CUDA:** CI/CPU only runs local toy-model regression tests. Qwen real model discovery needs CUDA; run `freeze.py --dry-run` to validate dataset without a model.
- **Flash Linear Attention backward:** inference working does not prove CUDA backward works. On kernel exceptions read `out/<run>/discovery/*_backward_traceback.txt`, include package versions and full traceback. Do not label all `RuntimeError`s as kernel errors.
- **`cannot import name extra from triton.experimental.gluon.language`:** earlier FLA environment required a compatible ~0.5.2 wheel; inspect your environment rather than blindly upgrading packages.
- **OOM on distribution semantic compilation:** the 248k×2560 embeddings can temporarily occupy several GB in float32; run only lexical compiler for smoke, compile distribution in a separate stage/run or reduce `--distribution-top` (the all-vocabulary similarity still computes for top-k).
- **`source changed since freeze`:** any change to root Python scripts invalidates the experiment identity. This is intentional; start a fresh run directory. Docs/tests changed after packaging do not enter root code hash but checksum manifest protects packaged file integrity.
- **`Discovery incomplete/unsealed`:** a previous GPU job failed before all candidates were stored. Re-run the *same* `discover.py` command to finish; differing resume parameters must use a new run.
- **`Discovery artifacts changed after sealing`:** any posthoc edit or truncated result file must be treated as a new experiment. Original sealed run is invalidated.
- **`stage manifest conflict`:** attempting a different top-k/dose/probe count in the same run violates preregistration. Create a new `--out` via `freeze.py`.
- **`missing *target.json` with `--cross-trait`:** discover all traits with identical compilers before asking for cross-trait validation.
- **No human labels:** generated text cannot be rated automatically by these scripts. `blind_review.csv` ratings must come from independent judges; numeric output scores alone cannot support expressive behavior claims.
- **Different p50 from prior Evil Pair reproduction:** expected! This package deliberately does **not** use p50. The standardized dose now maps to norm-1 direct MLP residual output, allowing a strict weight-only branch. `dose=0.5` has different meaning from the v4 `dose=0.05`.
- **Checkpoint data are not cryptographically secure from the machine owner:** SHA files/manifest identify inadvertent tampering and enable audit. Whoever can rewrite the entire run dir can also forge its internal checksums; archive the manifest hashes outside the execution host for high-integrity replication.
