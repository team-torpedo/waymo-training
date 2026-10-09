# Contributing

Thanks for helping Team Torpedo's pipeline. Quick rules:

1. **Branch** from `main`: `feature/<short-name>` or `fix/<short-name>`; open a pull request.
2. **Set up**: `bash scripts/setup_venv.sh --dev && source venv/bin/activate`.
3. **Before pushing**:
   ```bash
   python -m ruff check src stages tests main.py
   python -m pytest -q
   ```
4. **No data or weights in git.** `data/`, `outputs/`, `*.pt`, `*.keras`, `*.parquet` are gitignored — keep it that way.
5. **Configurability first.** New behaviour should be controlled from `config/*.yaml` and documented in
   `docs/configuration.md`. New swappable parts go in a registry (see `docs/extending.md`).
6. **Add a test** using the synthetic fixtures in `tests/conftest.py` for new data/model logic.
7. **Commit messages**: imperative, short subject (`Add focal loss`), details in the body if needed.
8. **Licensing**: contributions are under the repository's MIT license. Don't paste code with incompatible licenses
   and don't commit Waymo data (non-commercial license).
9. Update `CHANGELOG.md` for user-visible changes.
