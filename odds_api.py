"""
Fetch upcoming NCAAB game odds from The Odds API.

Docs: https://the-odds-api.com/liveapi/guides/v4/
Endpoint: GET /v4/sports/{sport}/odds/
"""

import requests
import pandas as pd

SPORT = "basketball_ncaab"
BASE_URL = "https://api.the-odds-api.com/v4/sports"


def fetch_upcoming_odds(
    api_key: str,
    regions: str = "us",
    markets: str = "totals,spreads",
    odds_format: str = "american",
    bookmakers: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch upcoming NCAAB game odds.

    Returns:
        (games_df, api_info)

        games_df columns:
            game_id, commence_time, home_team, away_team,
            total_point, over_price, under_price,
            spread_home_point, spread_home_price, spread_away_price,
            bookmaker

        api_info:
            requests_remaining, requests_used
    """
    url = f"{BASE_URL}/{SPORT}/odds/"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    api_info = {
        "requests_remaining": resp.headers.get("x-requests-remaining", "?"),
        "requests_used": resp.headers.get("x-requests-used", "?"),
    }

    events = resp.json()
    rows = []

    for event in events:
        game_id = event.get("id", "")
        commence = event.get("commence_time", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            total_point = None
            over_price = None
            under_price = None
            spread_home_point = None
            spread_home_price = None
            spread_away_price = None

            for market in bm.get("markets", []):
                if market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Over":
                            total_point = outcome.get("point")
                            over_price = outcome.get("price")
                        elif outcome["name"] == "Under":
                            under_price = outcome.get("price")

                elif market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            spread_home_point = outcome.get("point")
                            spread_home_price = outcome.get("price")
                        elif outcome["name"] == away:
                            spread_away_price = outcome.get("price")

            if total_point is not None:
                rows.append({
                    "game_id": game_id,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "total_point": total_point,
                    "over_price": over_price,
                    "under_price": under_price,
                    "spread_home_point": spread_home_point,
                    "spread_home_price": spread_home_price,
                    "spread_away_price": spread_away_price,
                    "bookmaker": bm_key,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        for col in ["total_point", "over_price", "under_price",
                     "spread_home_point", "spread_home_price", "spread_away_price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, api_info


def pick_consensus_line(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multi-bookmaker rows into one consensus row per game.

    Uses the median total across bookmakers. For spread, takes the median
    home spread. For prices, takes the first available bookmaker's prices
    (used for display only, not betting logic).
    """
    if odds_df.empty:
        return odds_df

    grouped = odds_df.groupby("game_id", sort=False)

    rows = []
    for game_id, grp in grouped:
        first = grp.iloc[0]
        rows.append({
            "game_id": game_id,
            "commence_time": first["commence_time"],
            "home_team": first["home_team"],
            "away_team": first["away_team"],
            "total_point": grp["total_point"].median(),
            "over_price": grp["over_price"].median(),
            "under_price": grp["under_price"].median(),
            "spread_home_point": grp["spread_home_point"].median(),
            "n_books": len(grp),
        })

    return pd.DataFrame(rows)
