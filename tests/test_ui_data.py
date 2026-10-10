"""The shared synthetic data dir and the app/components.py contract."""

from __future__ import annotations

import sys
from pathlib import Path

from fpl import storage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import components  # noqa: E402


def test_ui_data_tables(ui_data):
    for t in ("players", "teams", "fixtures", "my_team", "my_entry", "xp", "form",
              "fixture_ticker", "news", "top10k_picks", "top10k_entries", "ownership"):
        assert storage.exists(t, root=ui_data), t
    picks = storage.read_table("top10k_picks", 6, ui_data)
    assert picks.groupby("entry_id").size().eq(15).all()
    own = storage.read_table("ownership", 6, ui_data)
    assert own["delta_vs_prev_gw"].notna().any()


def test_components_contract(ui_data):
    sq = components.my_squad()
    assert len(sq) == 15
    assert {"web_name", "team_short", "position", "sell_price", "xp_gw7", "xp_next6"} <= set(sq.columns)
    assert sq["slot"].tolist() == list(range(1, 16))
    assert sq["xp_next6"].notna().all()
    assert "Haaland" in components.player_label(int(sq.loc[sq.web_name == "Haaland", "player_id"].iloc[0]))
    assert components.team_shorts()[1] == "ARS"
    assert components.team_colour("ARS") == components.team_colour(1) == "#EF0107"
    assert components.load("top10k_picks", 99).empty
