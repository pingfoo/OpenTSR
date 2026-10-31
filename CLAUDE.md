# CLAUDE.md

Guidance for Claude (Claude Code and claude.ai sessions) working in this repository.

## What this project is

OpenTSR builds an open-source, open-data traffic signal (traffic light) recognition model: detection plus state recognition, at commercial-grade quality. Everything is released openly: code, data, and trained weights. See README.md for the goal statement.

## Where work happens

- Local: `~/Developer/OpenTSR` on the maintainer's machine (Claude Code). GPU training and anything that needs lab hardware runs here or on lab servers.
- Cloud: claude.ai sessions in the Project "世界最高峰TSR". No GPU there; use it for research, scaffolding, docs, and light CPU checks.
- Both push to `github.com/pingfoo/OpenTSR`. The remote is the single source of truth; never rely on a session's local state surviving.

## Git conventions

- `main` is the integration branch and should always be in a working state.
- Before any push, run `git fetch origin main` and rebase (or merge) so the push lands on top of the latest remote.
- For a task that takes more than a couple of commits, work on a branch named after the topic (for example `survey-datasets`, `baseline-yolo`) and open a PR into `main`.
- Commit messages: English, imperative mood, a short subject line and a body that says why.
- Never commit data, weights, run outputs, or secrets. `.gitignore` already excludes the usual directories; keep it that way.

## Repository layout (planned; create directories as they become needed)

- `src/opentsr/`: library code (data loading, models, training, evaluation)
- `scripts/`: entry points (train, evaluate, export, data preparation)
- `configs/`: experiment and model configs; every reported result must map to a config
- `docs/`: surveys, design notes, and decisions (see below)
- `data/`, `weights/`, `runs/`: local only, gitignored; document where the public copies live

## Documentation

- Write code comments, docs, and commit messages in English. Conversation with the maintainer is in Japanese.
- Record decisions in `docs/decisions.md` as dated entries: what was decided, why, and the alternatives considered.
- Record survey results (datasets, OSS implementations, papers) under `docs/`, one file per topic.
- Keep `docs/third_party.md` listing every dataset, model, and codebase we build on, with its license and source URL. Only use assets whose license allows redistribution under Apache-2.0, or document the exception explicitly.
- Do not use em dashes in English prose; use periods, commas, or colons.

## Working principles

- Open by default: prefer public datasets and permissively licensed code. Anything we generate (synthetic data, labels, weights) is meant to be released.
- Reproducibility: pin dependencies, fix seeds, and make every number in a report reproducible from a config and a commit hash.
- Measure first: establish a fixed public benchmark and an evaluation script early, then improve against it.
