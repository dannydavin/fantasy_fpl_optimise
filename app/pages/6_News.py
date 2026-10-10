"""News feed and FPL injury flags, with my squad highlighted."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import components  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

HINT = "Run `uv run python scripts/update.py` to pull news."


def group_items(news: pd.DataFrame, my_ids: set[int]) -> pd.DataFrame:
    """Collapse (item, player) rows to one row per item; mine-first, then newest."""
    if news.empty:
        return pd.DataFrame(columns=[*news.columns, "players", "mine"])
    rows = []
    for item_id, g in news.groupby("item_id", sort=False):
        first = g.iloc[0]
        ids = {int(i) for i in g["player_id"].dropna()}
        names = [str(n) for n in dict.fromkeys(g["player_name"].dropna())]
        rows.append({
            "item_id": item_id, "published": first["published"], "source": first["source"],
            "kind": first["kind"], "title": first["title"], "summary": first["summary"],
            "url": first["url"], "players": ", ".join(names), "mine": bool(ids & my_ids),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["mine", "published"], ascending=[False, False],
                           na_position="last").reset_index(drop=True)


def filter_items(items: pd.DataFrame, sources, kinds, query: str, mine_only: bool) -> pd.DataFrame:
    m = items["source"].isin(sources) & items["kind"].isin(kinds)
    if query:
        text = (items["title"].fillna("") + " " + items["summary"].fillna("") + " "
                + items["players"].fillna(""))
        m &= text.str.lower().str.contains(query.lower(), regex=False)
    if mine_only:
        m &= items["mine"]
    return items[m]


def my_flags(squad: pd.DataFrame) -> pd.DataFrame:
    if squad.empty:
        return squad
    chance = squad["chance_of_playing_next_round"]
    return squad[(squad["status"].fillna("a") != "a") | (chance.notna() & (chance < 100))]


def rel_time(ts, now=None) -> str:
    if pd.isna(ts):
        return ""
    now = now or pd.Timestamp.now(tz="UTC")
    s = (now - pd.Timestamp(ts)).total_seconds()
    if s < 3600:
        return f"{max(int(s // 60), 0)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def _trunc(s, n=240) -> str:
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    st.set_page_config(page_title="News", layout="wide")
    components.reload_button()
    st.title("News")
    squad = components.my_squad()
    players = components.load("players")
    news = components.load("news")
    my_ids = {int(i) for i in squad["player_id"]} if not squad.empty else set()
    if news.empty and players.empty:
        st.info(f"No news data yet. {HINT}")
        return

    st.header("My squad flags")
    flags = my_flags(squad)
    if flags.empty:
        st.success("No injury or availability flags on my squad.")
    else:
        for r in flags.itertuples():
            chance = ("" if pd.isna(r.chance_of_playing_next_round)
                      else f" ({int(r.chance_of_playing_next_round)}% chance)")
            st.warning(f"**{r.web_name}** [{r.team_short}] status {r.status}{chance}: "
                       f"{r.news or 'no detail'}")

    st.header("Feed")
    items = group_items(news, my_ids)
    if items.empty:
        st.info(f"No news items. {HINT}")
    else:
        c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
        srcs = sorted(items["source"].dropna().unique())
        kinds = sorted(items["kind"].dropna().unique())
        sel_s = c1.multiselect("Source", srcs, default=srcs)
        sel_k = c2.multiselect("Kind", kinds, default=kinds)
        query = c3.text_input("Search")
        mine_only = c4.toggle("My players only")
        shown = filter_items(items, sel_s, sel_k, query, mine_only)
        st.caption(f"{len(shown)} of {len(items)} items")
        for r in shown.head(100).itertuples():
            title = f"[{r.title}]({r.url})" if isinstance(r.url, str) and r.url else r.title
            star = ":star: " if r.mine else ""
            st.markdown(f"{star}**{title}**  \n{r.source} · {r.kind} · {rel_time(r.published)}"
                        + (f" · players: {r.players}" if r.players else ""))
            if _trunc(r.summary):
                st.caption(_trunc(r.summary))

    st.header("League-wide flags")
    if players.empty:
        st.info(f"No players table. {HINT}")
        return
    fl = players[(players["status"] != "a") | (players["chance_of_playing_next_round"] < 100)]
    a, b = st.columns(2)
    sel_t = a.multiselect("Team", sorted(fl["team_short"].dropna().unique()), default=[])
    sel_p = b.multiselect("Position", ["GKP", "DEF", "MID", "FWD"], default=[])
    if sel_t:
        fl = fl[fl["team_short"].isin(sel_t)]
    if sel_p:
        fl = fl[fl["position"].isin(sel_p)]
    fl = fl.assign(mine=fl["player_id"].isin(my_ids)).sort_values(
        "selected_by_percent", ascending=False)
    st.dataframe(
        fl[["mine", "web_name", "team_short", "position", "status",
            "chance_of_playing_next_round", "selected_by_percent", "news", "news_added"]],
        hide_index=True, width="stretch",
        column_config={"selected_by_percent": st.column_config.NumberColumn(format="%.1f")})


if __name__ == "__main__":
    main()
