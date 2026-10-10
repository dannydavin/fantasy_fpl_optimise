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
| `--compare-chip CHIP` | Re-solve with CHIP in each GW of the horizon and print a table against holding it, plus the blank/double calendar. |

The script prints bank/FT/available chips, each GW's transfers (out → in with prices), an
FT/hit summary, the planning-GW captain/vice and XI xP, `net xP over horizon`, `legal` (or
the violations), and the gain versus making no transfers.

**Chip questions** ("is bench boost in GW9 worth it?"): use `--compare-chip bboost` (same
horizon/lock/ban as otherwise) rather than a single with/without pair. The user is really
asking "when should I play it", so the week they named only makes sense next to the other
weeks: a GW9 boost worth +13.5 means little if GW10 is worth +15.7. The printed calendar
shows doubles/blanks; a double gameweek is usually where a bench boost or triple captain
earns most, so if one exists inside or just beyond the horizon, say so. Differences of a
point or two between weeks are within model noise; say that rather than over-claiming.
Only chips listed as available can be played.

Read the `hold` row carefully: it means "not played inside this horizon", not "never played".
Every chip week beats it almost by construction, because the chip's value in a later week
isn't counted. So "every week beats hold" is not an argument to play now. The real question is
whether the best in-horizon week beats what the chip is likely worth later (a DGW bench boost
is often 25+ bench xP vs ~16-20 in a normal week). With no DGW in sight and the season
running on, playing it in the best week is reasonable; with a DGW likely later, hold.
If a row's `hits` differs from the others, part of its gain is the transfer, not the chip.

## 3. Sanity checks

- The planner must print `legal`. If it doesn't, report the violations rather than a plan.
- With a cross-check, compare the planning-GW move and captain. Later-GW differences are
  normal horizon-edge noise and not worth mentioning; a different move *this* week is.
- Check `status`/`news`/`chance_of_playing_next_round` (from
  `fpl.storage.read_table("players")`) for players coming in and the captain.

## 4. Report

The user wants short, scannable answers that lead with the decision. Report problems, not
the absence of them: mention an injury flag only if someone is flagged, and the cross-check
only as "agrees" or, if it disagrees on this week, both options. Use this shape:

```
**GW7** · £0.4m ITB · 1 FT · chips: FH, BB, TC

**Transfer:** Rogers (£7.7m) → Tavernier (£6.1m), no hit
**Captain:** Haaland · **Vice:** Gabriel
+11.5 xP over 5 GWs vs rolling · cross-check agrees

Plan by GW:
- GW7: Rogers → Tavernier
- GW8: Ødegaard → Mbeumo
- GW9: King → Le Fée
- GW10: Tzolakis → Trafford
- GW11: roll
```

If the recommendation is to roll the FT, say "**Transfer:** roll (1 → 2 FT)". If there's a hit, put
it on the transfer line. Add any lock/ban the user asked for as a one-liner so they can see it was
honoured.

For chip questions, lead with the recommendation (play in GW N / hold), then the table:

```
| Chip GW | Net xP | vs hold | Bench xP | Hits |
|---|---|---|---|---|
| hold | 305.96 | – | – | 0 |
| 9 | 319.50 | +13.5 | 16.5 | 0 |
| 10 | 321.70 | +15.7 | 18.2 | 0 |
```

then one or two lines on the calendar (doubles/blanks) and noise. Keep the plan-by-GW list
from the same run if it changes this week's move.

Never make transfers on the FPL website; the user does that. To browse everything:
`uv run streamlit run app/Home.py` (hit **Reload data** in the sidebar after a refresh).
