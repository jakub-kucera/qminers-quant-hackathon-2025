# originals/

The models as the team left them after the hackathon, moved here from `src/` (and Marek's
from his branch) with their history: `git log --follow originals/<file>` goes back to
November 2025. Apart from restoring one file to an earlier version (below), they are not edited:
`tests/test_originals.py` checks that each model in `qqh/` places exactly the same bets as its
original here.

| File | Model in `qqh.models` | Last changed | Notes |
|---|---|---|---|
| `model.py` | `ensemble` | 27 Nov 2025 | the ensemble; the others import its signals and subclass it |
| `model_hybrid_round_2.py` | `hybrid` | 6 Dec, 12:04 | most likely the finals' round-2 upload (see below) |
| `model_hybrid.py` | `hybrid-retuned` | 6 Dec, 15:07 | retuned after round 2; the last version on disk |
| `model_bookie_rookie.py` | `bookie-rookie` | 28 Nov | as of the Bookie Rookie task, before a finals-morning edit |
| `model_simple_rolling.py` | `simple-rolling` | 6 Dec, 15:20 | built on the finals day, never submitted |
| `model_underdog_sniper.py` | `underdog-sniper` | 6 Dec, 15:16 | Pepe's model, the finals' round-3 upload |
| `model_empty_zeros.py` | `no-bets` | 6 Dec, 12:22 | the finals' round-1 upload |
| `marek_model.py` | `market-residual` | 26 Nov | Marek's network, weights embedded |

The server took one file, so the uploads were these merged by hand with `model.py`;
`tools/bundle.py` now does that for the package's models. The exact uploads are still on the
evaluation server and are not here.

**Which hybrid went into round 2.** Two versions exist from the finals day. Replayed on
round 2's published data, the 12:04 one places no bets, which is what the server reported;
the 15:07 one places five. Hence the guess.

**Bookie Rookie.** `model_bookie_rookie.py` was edited on the finals morning to stop betting
above a bankroll of 1,400 (commit 8a0a5ca); the file here is the 28 November version, as for
the Bookie Rookie task.
