---
name: gameweek
description: Weekly FPL decision routine for team 1608168 in this repo — refresh data, run the MILP transfer planner (scripts/plan.py), cross-check with FPL-Optimization-Tools, and report transfers, captain/vice and chip value. Use this whenever the user asks anything about this week's or upcoming FPL moves, even casually — "who should I captain", "any transfers worth making", "should I bench boost / free hit / triple captain", "plan the next few GWs", "keep X, don't want Y", "is a -4 worth it", or a pre-deadline check.
---

# Gameweek routine

Run everything from the repo root with the Bash tool and prefix commands with
`PYTHONIOENCODING=utf8` — the Windows console garbles names like Ødegaard/Groß otherwise.

## 1. Refresh

`PYTHONIOENCODING=utf8 uv run python scripts/update.py` (~15 s, light on the API). Step
failures are caught and printed; a Reddit 403 is expected and harmless. Only add
`--top10k --limit 500` if the user wants crowd/ownership data refreshed; the full
`--limit 10000` pull takes ~40 min of API calls, so run it only when explicitly asked.

## 2. Plan

`PYTHONIOENCODING=utf8 uv run python scripts/plan.py [options]`

| Option | Use |
|---|---|
| `--horizon N` | GWs to plan over (1–6, default 5). Match what the user asks ("next 6 GWs" → 6). |
| `--cross-check` | Also run FPL-Optimization-Tools (~1–2 min). Default on for a weekly check; skip if the user says so. |
| `--lock NAME/ID ...` | Players to keep (or force in). Names are accent/case-insensitive web names, e.g. `--lock Haaland "Joao Pedro"`. |
| `--ban NAME/ID ...` | Players never to own. |
| `--chip GW:chip` | Play a chip: `wildcard`, `freehit`, `bboost`, `3xc`. Repeatable. |

The script prints the bank/FT/available chips, each GW's transfers (out → in with prices),
an FT/hit summary table, the planning-GW captain/vice and XI xP, `net xP over horizon`,
`legal` (or the violations), and the gain versus making no transfers.

**Chip questions** ("is bench boost in GW9 worth it?"): one run can't answer this, so run
the planner twice with identical options, once with `--chip GW:chip` and once without, and
compare `net xP over horizon`. The difference is the chip's value under this plan. Mention
that holding the chip keeps it for a later, possibly better week (DGWs, see the Fixtures
page), so a small gain is a weak reason to play it now. Only chips listed as available can
be played.

## 3. Sanity checks

- The planner must print `legal`. If it doesn't, report the violations rather than a plan.
- With a cross-check, compare the planning-GW move and captain. Disagreement in later GWs
  is normal (it's near the horizon edge); disagreement on this week's move is worth flagging
  with both options.
- For each player coming in and the captain, check injury/doubt status:
  `fpl.storage.read_table("players")` columns `status`/`news`/`chance_of_playing_next_round`.
  The xP model already discounts flagged players, but the user wants to hear about them.

## 4. Report

Keep it short and lead with the decision:

- GW, bank, FTs, chips available
- This week's transfer(s): out → in, prices, any hit — or "roll the FT" if none
- Captain / vice
- Gain vs no transfers over the horizon (and chip value, if asked)
- Whether the cross-check agrees
- Later-GW moves in one line
- Any injury/doubt flags

Never make transfers on the FPL website; the user does that. To browse everything:
`uv run streamlit run app/Home.py` (hit **Reload data** in the sidebar after a refresh).
