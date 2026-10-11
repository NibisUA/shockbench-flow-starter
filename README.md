# Merge-Conflict: ShockBench-Flow

## Основна фінальна версія

Основний агент команди **Merge-Conflict** — **team_agent_context_dispatch_final**.
Готовий ZIP і його вихідний код нижче відповідають одній і тій самій перевіреній версії.

- [Завантажити фінальний ZIP](https://github.com/NikitaBil/shockbench-flow-starter/raw/refs/heads/main/submissions/team_agent_context_dispatch_final.zip)
- [Вихідний код фінального агента](agents/team_agent_context_dispatch_final/)
- [Метадані версії та перевірок](submissions/FINAL_AGENT.json)

SHA256 ZIP:

```text
7be1c2f5631f8a582c161da48726089735737dc7c7bed9ae6675331c74901e81
```

### Результати

**Codabench RSS: 0.7326**, за результатом, повідомленим командою. Ця публікація не виконує нового сабміту на Codabench.

Наведені нижче оцінки отримано локально на самому фінальному ZIP, окремо від результату Codabench:

| Набір | Епізоди | RSS | 90% інтервал RSS | Приріст до v30 |
| --- | ---: | ---: | --- | ---: |
| Small, entropy 20261073 | 64 | 0.731704 | [0.705541, 0.757897] | +0.000995 |
| Full, entropy 20261071 | 40 | 0.791635 | [0.764436, 0.816749] | 0.000000 |

Парний 90% інтервал приросту Small: [0.000587, 0.001455]. На Full всі 40 витрат точно збігаються з v30.
Локальні перевірки Small і Full пройшли з нульовим fallback; максимальний CPU-час тижня становив 0.446 с і 0.860 с відповідно.
Ці таймінги отримано на локальній машині, вони не сертифікують швидкість на машині організаторів.

### Як перевірити та запустити

Після `uv sync`:

```bash
uv run sbf check team_agent_context_dispatch_final --task=small
uv run sbf check team_agent_context_dispatch_final --task=full
uv run sbf evaluate team_agent_context_dispatch_final --task=small
uv run sbf evaluate team_agent_context_dispatch_final --task=full
```

Агент читає розміри мережі з config. Для Small він використовує облік звільнення складів вихідними відправками з v35; для Full зберігає перевірену поведінку v30.
Активні механізми включають економічний розподіл запасів, LP продажів, облік затримок і черг, захист від переповнення, паливні партії та узгодження нафти.
OilPulseAlignment порівнює два варіанти поставки нафти на два тижні за публічними даними.
Прогноз ризику використовує фіксовані коефіцієнти та історію поточного епізоду. Автоматичне онлайн-навчання на топ-доставках і помилках є напрямом розвитку.

## Документація starter kit

**A Gymnasium control task: keep a supply network running while the map breaks.**

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-1.3-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Every week of an episode you decide how much of each good to send along each route of a network: fuel to power grids,
wafers to chip factories, chips to markets. Before the episode starts, random disruptions are drawn: a strait closes, a
route is sanctioned, a tariff jumps, a factory goes down. You see the network as it is this week, your stock, a demand
forecast, and noisy early warnings. An episode costs money (freight, tariffs, storage, unmet demand). Lower is better.

Your **score** compares your cost with two references on the same scenarios: **0** is the naive rule (keep shipping
the normal plan), **1** is the clairvoyant plan (it knew every disruption in advance). Below 0 is worse than naive. You
do not need to know supply chains: treat it as a regular gym task and try RL, control, search or evolved policies.

![The Small network (the public board's)](docs/img/network_small.png)

## Installation and usage

### 1. Install uv

[uv](https://docs.astral.sh/uv/) installs Python and this project's packages, in place of pip and venv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For alternatives (Windows, Homebrew, pip), see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).
You also need [git](https://git-scm.com/downloads).

### 2. Fork and clone the repository

Fork this repository on GitHub (the **Fork** button), then clone your fork:

```bash
git clone <your fork's URL> shockbench-flow-starter
cd shockbench-flow-starter
```

### 3. Create and activate the virtual environment

```bash
uv sync
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate`. `uv sync` downloads Python 3.13 if you do not have it, and installs
the exact package versions of `uv.lock` into `.venv/`. For the PPO example, run `uv sync --extra rl` instead: it adds
Stable-Baselines3 and PyTorch.

The commands below assume the environment is active. Without it, put `uv run` in front: `uv run sbf evaluate mine`.

### 4. Run the quick start

```bash
python examples/01_quickstart.py
```

It plays one episode of the practice network (Tiny) with random actions and prints the cost. The first run is slow
while Python compiles the packages.

### 5. Create your agent

```bash
cp -r agents/template agents/mine
```

Then edit `act()` in `agents/mine/agent.py`. An agent is one class:

```python
import numpy as np

class Agent:
    def __init__(self, config=None):   # once per episode
        u0 = config["static"]["edges"]["u0"]
        self.cap = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]])

    def act(self, observation):        # once per week
        return {"flows": self.cap * observation["action_mask"]}  # send the maximum on every allowed route
```

On the server, `agent.py` may only import Python's standard library, numpy, SciPy and PyTorch (CPU). Put the files it
loads (weights, tables) in its folder.

### 6. Score it

```bash
sbf evaluate mine                # your score on Tiny's 20 dev episodes, with a 90 % interval
sbf compare mine template        # did your change help? a paired interval
```

The first run on a network computes the reference costs and caches them: about a minute on Tiny, longer on Small and
Full. When your agent works on Tiny, add `--task=small`: Small is the network the public board scores.

### 7. Check and submit

Register for the competition on Codabench and wait for the organisers' approval. Then:

```bash
sbf check mine --task=small      # the server's checks and a timed run
cp .env.example .env             # then set CODABENCH_COMPETITION in .env to the competition's URL
sbf token                        # once: asks for your Codabench login and saves your API token in .env
sbf upload mine --dry_run        # checks everything, uploads nothing
sbf upload mine --wait           # submits, then waits for the score
```

- Each upload uses one of your 3 daily submissions. `sbf upload` never retries.
- `sbf token` exists because Codabench's pages do not show your API token. An account made with "Sign in with
  GitHub" needs a password first (Codabench's password reset).
- `sbf upload` uses Codabench's undocumented web API, which may change. The competition page always works: upload
  `outputs/mine.zip`, which `sbf pack mine` writes.

## The task

| Environment           | Role                            | Nodes | Edges | Goods | Straits | Weeks | `flows` slots |
| --------------------- | ------------------------------- | ----- | ----- | ----- | ------- | ----- | ------------- |
| `ShockBench/Tiny-v0`  | practice, the default           | 12    | 25    | 4     | 1       | 26    | 20            |
| `ShockBench/Small-v0` | **the public board**            | 38    | 115   | 8     | 7       | 52    | 108           |
| `ShockBench/Full-v0`  | **the private board**           | 72    | 369   | 8     | 7       | 104   | 395           |

- **Action**, a dict of arrays: `flows` (how much to send on each route this week, 0 or more), and optionally
  `override_qty` and `release_mode` (how tankers queued at a strait leave: 0 the default rule, 1 your quantities,
  2 hold).
- **Observation**, a dict of arrays keyed by name: stock, backlog, goods in transit, the network this week (capacity,
  cost, lead time, sanctions and tariffs, how open each strait is), last week's costs, the demand forecast, warnings
  and announcements. Each field `x` has a mask `x.observed`; `action_mask` marks the routes you may use this week.
- **Reward**: minus the week's cost in USD. Weeks cost millions to billions, so for RL use the `ScaleReward` wrapper.

Read every shape from the agent's `config`: the networks differ in size. [docs/GUIDE.md](docs/GUIDE.md) has the
details, and [docs/fields/](docs/fields/) lists every field of each network.

## Commands

| Command                | What it does                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `sbf evaluate <agent>` | Your score on the dev episodes, its 90 % interval, the costs in USD, and the weeks the naive rule played for you.   |
| `sbf compare <a> <b>`  | Both agents on the same episodes. If the interval of the difference holds 0, the episodes cannot tell them apart. |
| `sbf check <agent>`    | The server's zip checks, every file's imports, and a timed run with only the server's packages (`--docker`: in a local copy of the scoring container). |
| `sbf pack <agent>`     | Writes `outputs/<name>.zip`, for the competition page's upload button.                                             |
| `sbf token`            | Gets your Codabench API token from your login and saves it in `.env`.                                              |
| `sbf upload <agent>`   | Packs the agent and submits it (`--dry_run` to rehearse, `--wait` for the score).                                   |
| `sbf status [id]`      | Your submissions and their scores.                                                                                  |

`<agent>` is a name (`mine` means `agents/mine/`), a folder or a zip. Every command takes `--task=tiny|small|full`
(default `tiny`); `sbf <command> --help` lists the rest. `--quick` gives a rough score in seconds (not the board's
numbers). To switch a flag off, write `--noquick` or `--quick=False`: lowercase `--quick=false` counts as on.

## Examples

Each example is one file: read it, copy it, change it. They run on Tiny by default and take flags like `sbf`
(`--help` lists them). Files they write go to `outputs/<example>/<date_time>/`. An agent that 05 trains or 06 searches
fits only the network it was made on: use `--task=small` for one you will submit.

| Example                                                 | What it shows                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [01_quickstart.py](examples/01_quickstart.py)           | the gymnasium loop with random actions                                                     |
| [02_play_agents.py](examples/02_play_agents.py)         | agents played under gymnasium, as the scorer plays them                                    |
| [03_heuristic_agent.py](examples/03_heuristic_agent.py) | a rule that reacts to strait closures, against send-the-maximum                            |
| [04_evaluate.py](examples/04_evaluate.py)               | `sbf evaluate` and `sbf compare` from Python                                               |
| [05_train_ppo.py](examples/05_train_ppo.py)             | PPO (Stable-Baselines3), exported as a submission the server can run (`uv sync --extra rl`) |
| [06_policy_search.py](examples/06_policy_search.py)     | an evolutionary search over an agent's numbers, and where an LLM proposer fits             |
| [07_dashboard.py](examples/07_dashboard.py)             | the network map, episode dashboards and a GIF                                              |

```bash
python examples/03_heuristic_agent.py --task=small --episodes=6
```

![An episode of send-the-maximum on Tiny, with naive on the same scenario](docs/img/dashboard_tiny.png)

## Glossary

- **Naive rule** (score 0): keep shipping the normal plan and ignore disruptions. A week your agent cannot play (an
  exception, a malformed action, over the CPU budget) is played by the naive rule instead.
- **Clairvoyant plan** (score 1; "the oracle" in the package's code): a linear program that knows the whole scenario in
  advance. No agent can beat it on average.
- **Score** ("RSS" in the package and on the board): the share of the saving from naive to the clairvoyant plan that
  your agent achieves.
- **Harm level** (1 calmest to 4 most harmful): how much damage an episode's disruptions would do. The board weights
  them 50 %, 30 %, 15 %, 5 %.
- **Dev split**: the 20 public episodes `sbf evaluate` uses by default. The boards score private episodes drawn the
  same way.
- **Root** (`entropy`): the seed of a set of scenarios. 0 is the dev split's; any other integer gives scenarios of your
  own to train on.

## Rules

The rules that decide a score (the server's packages, CPU budgets, seeding, size limits, the boards, submission limits)
are in [docs/GUIDE.md](docs/GUIDE.md#rules). The dates are on the competition page.

## Layout

It is your fork: change anything.

```
agents/            one folder per agent: template (send the maximum), random, heuristic, and yours
examples/          01_quickstart.py ... 07_dashboard.py, and ppo_agent.py (the PPO submission's agent.py)
src/sbf_starter/   the `sbf` command line
docs/              GUIDE.md (interface, rules, scoring) and fields/ (every observation and action field)
scripts/           fields_docs.py: regenerates docs/fields/ after a new shockbench-flow release
tests/             pytest -n 3
outputs/           run folders and zips (gitignored)
```

The benchmark itself is the [`shockbench-flow`](https://pypi.org/project/shockbench-flow/) package from PyPI.
Add a package with `uv add <package>`; `agent.py` still may only import the server's packages.
[AGENTS.md](AGENTS.md) is the same map for a coding assistant.

## Updating

`pyproject.toml` asks for `shockbench-flow >= 0.1.2`; `uv.lock` records the exact version everyone installs. Update
with uv, never by editing a version:

```bash
uv tree --outdated --depth 1                  # is there a newer shockbench-flow? it shows "(latest: ...)"
uv sync --upgrade-package shockbench-flow     # install the newest one; your other packages stay as locked
```

Add `--extra rl` to `uv sync` if you use PPO: without it, uv removes Stable-Baselines3 and PyTorch. Commit the
changed `uv.lock`. Fixes to the kit itself come from this repository:

```bash
git remote add upstream <this repository's URL>    # once
git pull upstream main
uv sync
```

If `uv.lock` conflicts because you added packages, take the new lock and relock:
`git checkout --theirs uv.lock && uv lock && uv sync`.

## FAQ

**Do I need to know supply chains or RL?** No. It is an episodic control problem with a dict observation and a dict
action. A good first step is to beat the template on Small.

**Which network is scored?** The public board plays Small, the private board Full ([docs/GUIDE.md](docs/GUIDE.md#rules)).

**How do I avoid overfitting the 20 dev episodes?** Train and tune on your own root
(`sbf evaluate mine --entropy=12345 --episodes=64`, or `gym.make(..., entropy=12345)`), and confirm on the dev
episodes with `sbf compare`.

**`sbf: command not found`.** Activate the environment (step 3), or put `uv run` in front. If you moved or renamed
the folder after `uv sync`, the environment still points at the old path: recreate it with
`deactivate; rm -rf .venv && uv sync && source .venv/bin/activate`.

**My agent crashes on Small but not on Tiny.** Read shapes from `config["spaces"]`. On Small and Full the queue at the
straits is one dense table (`queue_lots.qty`); Tiny's per-lot lists do not exist there
([docs/GUIDE.md](docs/GUIDE.md#small-and-full)).

**Where do I ask questions?** On the competition's forum.

## Licence

MIT, see [LICENSE](LICENSE). The shockbench-flow package is MIT too; its licence and third-party notices are in the
package, under `shockbench_flow-*.dist-info/licenses/`.
