# CLAUDE.md

Guidance for Claude (Claude Code and claude.ai sessions) working in this repository.

## What this project is

OpenTSR builds an open-source, open-data traffic sign recognition model (detection plus
classification) at commercial-grade quality. Everything is released openly: code, data, and
trained weights. See README.md for the goal statement.

Traffic lights were the original target; that work is kept as a secondary track under
`traffic_lights/` (see below). Work on traffic signs first unless the maintainer asks for
traffic lights.

## Where work happens

- Local: `~/Developer/OpenTSR` on the maintainer's machine (Claude Code). GPU training and anything that needs lab hardware runs here or on lab servers.
- Cloud: claude.ai sessions in the Project "世界最高峰TSR". No GPU there; use it for research, scaffolding, docs, and light CPU checks.
- Both push to `github.com/pingfoo/OpenTSR`. The remote is the single source of truth; never rely on a session's local state surviving.

## Git conventions

- `main` is the integration branch and should always be in a working state.
- Before any push, run `git fetch origin main` and rebase (or merge) so the push lands on top of the latest remote.
- For a task that takes more than a couple of commits, work on a branch named after the topic (for example `survey-datasets`, `baseline-yolo`) and open a PR into `main`.
- Commit messages: English, imperative mood, a short subject line and a body that says why.
- Never commit data, weights, run outputs, or secrets. `.gitignore` already excludes the usual directories (anchored to the root, so a package named `datasets/` is still tracked); keep it that way and check `git status` for files that should have been committed.

## Repository layout (create directories as they become needed)

The root belongs to traffic signs. Everything specific to traffic lights sits under a
`traffic_lights` path segment and nowhere else; never put traffic light files in the root
`docs/`, `configs/` or `scripts/`, and never put sign files under `traffic_lights`.

- `src/opentsr/`: shared library code used by both tracks (`metrics.py`, `splits.py`). Shared
  modules and `opentsr.signs` must not import from `opentsr.traffic_lights`.
- `src/opentsr/signs/`: traffic sign library code (data loading, models, training)
- `scripts/`, `configs/`, `docs/`: traffic sign entry points, configs, surveys; `docs/` also holds
  the project-wide `decisions.md` and `third_party.md`
- `traffic_lights/`: the whole traffic light track (`docs/`, `configs/`, `scripts/`, `Makefile`)
- `src/opentsr/traffic_lights/`: traffic light library code
- `data/`, `weights/`, `runs/`: local only, gitignored; traffic light files go in their
  `traffic_lights/` subdirectory (for example `data/traffic_lights/s2tld`). Document where the
  public copies live.
- `Makefile`: sign targets plus `make lights-<target>`, which runs `traffic_lights/Makefile`

## Documentation

- Write code comments, docs, and commit messages in English. Conversation with the maintainer is in Japanese.
- Record decisions as dated entries (what was decided, why, and the alternatives considered): project-wide and sign decisions in `docs/decisions.md`, traffic-light-only decisions in `traffic_lights/docs/decisions.md`.
- Record survey results (datasets, OSS implementations, papers) one file per topic: under `docs/` for signs, under `traffic_lights/docs/` for traffic lights.
- Keep `docs/third_party.md` (shared and sign assets) and `traffic_lights/docs/third_party.md` (traffic light assets) listing every dataset, model, and codebase we build on, with its license and source URL. Only use assets whose license allows redistribution under Apache-2.0, or document the exception explicitly.
- Do not use em dashes in English prose; use periods, commas, or colons.

## Working principles

- Open by default: prefer public datasets and permissively licensed code. Anything we generate (synthetic data, labels, weights) is meant to be released.
- Reproducibility: pin dependencies, fix seeds, and make every number in a report reproducible from a config and a commit hash.
- Measure first: establish a fixed public benchmark and an evaluation script early, then improve against it.
