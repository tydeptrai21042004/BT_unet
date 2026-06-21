# Repository cleanliness

The source package must not contain Python bytecode, cache directories,
checkpoints, datasets, archives, generated media, or other binary artifacts.

Preview removable artifacts:

```bash
python scripts/clean_repository_artifacts.py
```

Remove them:

```bash
python scripts/clean_repository_artifacts.py --apply
```

Verify the source tree:

```bash
python tools/audit_repository_cleanliness.py
```

The `.gitignore` also blocks common cache, model-weight, dataset, output,
archive, and generated-media files from being committed again.
