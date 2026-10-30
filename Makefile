# neuron-closed — one entry point for every claim the docs make.
#
#   make check     re-derive all checks (doc blocks, zip<->tree, checksums, compile, tests, dedup, git)
#   make meta      re-render the two auto blocks in README.md from doc/metadata.json / packages
#   make hooks     opt in to the repo pre-commit hook (git config core.hooksPath .githooks)
#
# `make check` is stdlib-only: no network, no GPU, no torch. Torch-dependent tests are
# reported as skipped; run them on the GPU box with `python -m unittest discover -s tests`.

PYTHON ?= python3
CHECK  := $(PYTHON) tools/check.py

.PHONY: help check blocks ziptrees checksums compile tests dedup git meta meta-check hooks

help:
	@sed -n '2,9p' Makefile

check:
	@$(CHECK) all

blocks:      ; @$(CHECK) blocks
ziptrees:    ; @$(CHECK) ziptrees
checksums:   ; @$(CHECK) checksums
compile:     ; @$(CHECK) compile
tests:       ; @$(CHECK) tests
dedup:       ; @$(CHECK) dedup
git:         ; @$(CHECK) git

meta:
	@$(CHECK) render-status --write
	@$(CHECK) render-provenance --write

meta-check:
	@$(CHECK) blocks

hooks:
	@git config core.hooksPath .githooks && echo "hooks enabled: .githooks (pre-commit and pre-push run make check)"

certify:
	bash tools/certify.sh
