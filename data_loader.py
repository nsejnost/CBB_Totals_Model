"""
Data loader for Barttorvik game-level stats and Vegas closing lines.

Loads local Barttorvik super_sked CSVs (2025, 2026) and merges with
the local Vegas closing-line dataset. Uses the curated team_name_mapping.csv
for reliable name matching between data sources.
"""

import os
import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Barttorvik column names (positional – the CSV has no header row)
# ---------------------------------------------------------------------------
TORVIK_COLS = [
    "muid", "date", "conmatch", "matchup", "prediction", "ttq", "conf",
    "venue",                                                        # 0-7
    "team1", "t1oe", "t1de", "t1py", "t1wp", "t1propt",
    "team2", "t2oe", "t2de", "t2py", "t2wp",                       # 8-18
    "t2propt", "tpro", "t1qual", "t2qual", "gp", "result",
    "tempo", "possessions", "t1pts",                                # 19-27
    "t2pts", "winner", "loser", "t1adjt", "t2adjt", "t1adjo",
    "t1adjd", "t2adjo", "t2adjd",                                  # 28-36
    "gamevalue", "mismatch", "blowout", "t1elite", "t2elite",
    "ord_date", "t1ppp", "t2ppp", "gameppp",                       # 37-45
    "t1rk", "t2rk", "t1gs", "t2gs", "gamestats", "overtimes",
    "t1fun", "t2fun", "results",                                   # 46-54
]

NUMERIC_TORVIK = [
    "t1oe", "t1de", "t1py", "t1wp", "t1propt",
    "t2oe", "t2de", "t2py", "t2wp", "t2propt",
    "tpro", "t1qual", "t2qual",
    "tempo", "possessions", "t1pts", "t2pts",
    "t1adjt", "t2adjt", "t1adjo", "t1adjd", "t2adjo", "t2adjd",
    "gamevalue", "mismatch", "t1ppp", "t2ppp", "gameppp",
    "t1rk", "t2rk",
]


# ---------------------------------------------------------------------------
# Local file loader for Barttorvik data
# ---------------------------------------------------------------------------

def _load_torvik_csv(path: str, season: int) -> pd.DataFrame:
    """Load a single Barttorvik super_sked CSV (headerless, positional cols)."""
    df = pd.read_csv(path, header=None, quotechar='"', on_bad_lines="skip")

    # Assign column names based on available columns
    n_cols = min(len(TORVIK_COLS), df.shape[1])
    df.columns = TORVIK_COLS[:n_cols]

    # Pad missing columns
    for c in TORVIK_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numerics
    for c in NUMERIC_TORVIK:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse dates: Torvik uses "M/D/YY" format (e.g. "11/4/24")
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df["season"] = season

    # Drop rows where we can't parse essential fields
    df = df.dropna(subset=["date", "team1", "team2", "t1pts", "t2pts"])

    return df


def load_all_torvik(base_dir: str | None = None) -> pd.DataFrame:
    """Load all local Barttorvik CSVs (2025 and 2026 seasons)."""
    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    frames = []
    for season in [2025, 2026]:
        path = os.path.join(base_dir, f"{season}_super_sked.csv")
        if os.path.exists(path):
            df = _load_torvik_csv(path, season)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=TORVIK_COLS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Vegas data loader
# ---------------------------------------------------------------------------

def load_vegas_data(path: str | None = None) -> pd.DataFrame:
    """Load the local CSV of closing spreads / totals."""
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__),
            "ncaab_proxy_close_spreads_totals_2024-11-04_to_2026-02-12.csv",
        )
    df = pd.read_csv(path)
    df["game_date_utc"] = pd.to_datetime(df["game_date_utc"])
    for c in [
        "closing_total_point",
        "closing_total_over_price",
        "closing_total_under_price",
        "closing_spread_home_point",
        "closing_spread_home_price",
        "closing_spread_away_point",
        "closing_spread_away_price",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Team-name matching using the curated mapping file
# ---------------------------------------------------------------------------

# Comprehensive mascot list for stripping from Vegas team names
# Sorted longest-first for greedy matching
_MASCOTS = sorted([
    "49ers", "aces", "aggies", "anteaters", "antelopes", "aztecs",
    "badgers", "banana slugs", "battlin bears", "beach", "bearcats",
    "bears", "beavers", "bengals", "big green", "big red", "billikens",
    "bison", "black bears", "black knights", "blazers", "blue demons",
    "blue devils", "blue hens", "blue jays", "blue raiders", "bluejays",
    "bobcats", "boilermakers", "braves", "broncos", "broncs", "bruins",
    "buccaneers", "buckeyes", "buffaloes", "bulldogs", "bulls",
    "camels", "campus", "cardinals", "catamounts", "cavaliers",
    "chanticleers", "chippewas", "colonels", "colonials", "commodores",
    "cornhuskers", "cougars", "cowboys", "crimson", "crimson tide",
    "crusaders", "cyclones", "deacons", "demon deacons", "dolphins",
    "dons", "doves", "dragons", "dukes", "dutchmen", "eagles",
    "explorers", "falcons", "fighting camels", "fighting hawks",
    "fighting illini", "fighting irish", "flames", "flash", "flyers",
    "friars", "gaels", "gamecocks", "gators", "generals", "golden bears",
    "golden eagles", "golden flash", "golden flashes", "golden gophers",
    "golden griffins", "golden grizzlies", "golden hurricane",
    "golden knights", "golden lions", "golden panthers", "gophers",
    "gorillas", "governors", "great danes", "green wave", "griffins",
    "grizzlies", "guards", "hawkeyes", "hawks", "highlanders",
    "hilltoppers", "hokies", "hornets", "hoosiers", "hoyas", "huskies",
    "hurricanes", "ichabods", "illini", "indians", "irish", "islanders",
    "jackrabbits", "jacks", "jaguars", "jaspers", "javelinas",
    "jayhawks", "jimmies", "judges", "kangaroos", "keydets", "knights",
    "koalas", "lakers", "lancers", "leathernecks", "leopards", "lion",
    "lions", "lobos", "longhorns", "lumberjacks", "mastodons",
    "matadors", "mavericks", "mean green", "midshipmen", "miners",
    "minutemen", "moccasins", "mocs", "monarchs", "mounties",
    "mountain hawks", "mountaineers", "musketeers", "mustangs",
    "nittany lions", "norse", "oaks", "orange", "orangemen", "owls",
    "paladins", "panthers", "patriarchs", "patriots", "peacocks",
    "penguins", "phoenix", "pilots", "pioneers", "pirates",
    "plainsmen", "poets", "pointers", "polar bears", "pride",
    "privateers", "purple aces", "purple eagles", "quakers", "racers",
    "raiders", "rainbow warriors", "rams", "rattlers", "razorbacks",
    "rebels", "red flash", "red foxes", "red hawks", "red raiders",
    "red storm", "red wolves", "redhawks", "redbirds", "retrievers",
    "revolutionaries", "river hawks", "roadrunners", "rockets", "roos",
    "royals", "running rebels", "saints", "salukis", "scarlet knights",
    "scots", "seahawks", "seawolves", "seminoles", "settlers",
    "shockers", "skyhawks", "sooners", "spartans", "spiders", "stags",
    "stallions", "statesmen", "stormy petrels", "sun devils",
    "sycamores", "tar heels", "terrapins", "terriers", "texans",
    "thunderbirds", "thundering herd", "tidal wave", "tigers", "titans",
    "toads", "toreadors", "toreros", "tornado", "tribe", "tritons",
    "trojans", "tropics", "vandals", "vikings", "volunteers", "voyagers",
    "vulcans", "warriors", "wasps", "wave", "wildcats", "wolf pack",
    "wolfpack", "wolverines", "wolves", "wombats", "wren",
    "yellow jackets", "zips",
], key=lambda s: -len(s))


def _strip_mascot(name: str) -> str:
    """Strip mascot suffix from a Vegas-style team name."""
    s = name.strip()
    s_lower = s.lower()
    for mascot in _MASCOTS:
        if s_lower.endswith(" " + mascot):
            stripped = s[:len(s) - len(mascot)].strip()
            if stripped:
                return stripped
    return s


def _normalize_school(name: str) -> str:
    """
    Normalize a school name for matching: expand abbreviations, standardize
    punctuation, etc.

    'Alabama St' -> 'alabama state'
    'Boston Univ.' -> 'boston university'
    'CSU Bakersfield' -> 'cal state bakersfield'
    """
    s = name.strip().lower()
    s = s.replace("'", "'")  # normalize quotes

    # Special full-name replacements (before general rules)
    special = {
        "gw": "george washington",
        "gwu": "george washington",
        "smu": "smu",
        "usc": "usc",
        "ucf": "ucf",
        "vcu": "vcu",
        "fiu": "fiu",
        "fdu": "fairleigh dickinson",
        "liu": "liu",
        "unc": "north carolina",
        "umbc": "umbc",
        "unlv": "unlv",
        "utep": "utep",
        "utsa": "utsa",
        "army": "army west point",
        "army knights": "army west point",
        "navy": "navy",
        "ole miss": "ole miss",
        "miami (oh)": "miami (oh)",
        "miami (fl)": "miami (fl)",
        "nc state": "nc state",
    }
    if s in special:
        return special[s]

    # CSU -> Cal State
    s = re.sub(r"^csu\b", "cal state", s)

    # Expand 'St' → 'State' (but not 'St.' which means Saint)
    # 'Alabama St' → 'Alabama State' but 'St. Francis' stays
    s = re.sub(r"\bst$", "state", s)  # "St" at end of string
    s = re.sub(r"\bst\b(?!\.)", "state", s)  # "St" not followed by "."

    # 'Univ.' or 'Univ' → 'University'
    s = re.sub(r"\buniv\.?\b", "university", s)

    # Standardise punctuation for final comparison
    s = s.replace(".", "").replace("'", "").replace("-", " ").replace("(", " (")
    s = re.sub(r"\s+", " ", s).strip()

    return s


def load_team_mapping(path: str | None = None) -> pd.DataFrame:
    """Load the curated team name mapping file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "team_name_mapping.csv")
    df = pd.read_csv(path)
    # Drop rows with NaN barttorvik_name
    df = df.dropna(subset=["barttorvik_name"])
    return df


def build_name_map(
    vegas_names: list[str],
    torvik_names: list[str],
    mapping_path: str | None = None,
) -> dict:
    """
    Build a mapping from Vegas team names -> Torvik team names.

    Uses the curated mapping file as primary source, with fuzzy fallback.
    """
    from difflib import get_close_matches

    mapping_df = load_team_mapping(mapping_path)

    # Build lookup: normalized(odds_api_name) -> barttorvik_name
    odds_to_torvik: dict[str, str] = {}
    for _, row in mapping_df.iterrows():
        api_norm = _normalize_school(str(row["odds_api_name"]))
        odds_to_torvik[api_norm] = str(row["barttorvik_name"])

    # Also build a normalized Torvik lookup for fuzzy fallback
    norm_to_torvik: dict[str, str] = {}
    for tn in torvik_names:
        tn_s = str(tn).strip()
        norm_to_torvik[tn_s.lower()] = tn_s

    torvik_lower_list = list(norm_to_torvik.keys())

    result: dict[str, str] = {}

    for vn in vegas_names:
        # Step 1: strip mascot
        stripped = _strip_mascot(vn)
        stripped_norm = _normalize_school(stripped)

        # Step 2: look up in mapping file
        if stripped_norm in odds_to_torvik:
            torvik_name = odds_to_torvik[stripped_norm]
            # Verify it exists in actual Torvik data
            if torvik_name.lower() in norm_to_torvik:
                result[vn] = norm_to_torvik[torvik_name.lower()]
                continue
            # Try case-insensitive match
            matches = get_close_matches(
                torvik_name.lower(), torvik_lower_list, n=1, cutoff=0.85
            )
            if matches:
                result[vn] = norm_to_torvik[matches[0]]
                continue

        # Step 3: try direct Torvik name match on stripped name
        if stripped.lower() in norm_to_torvik:
            result[vn] = norm_to_torvik[stripped.lower()]
            continue

        # Step 4: try normalized stripped name against Torvik names
        matches = get_close_matches(
            stripped_norm, torvik_lower_list, n=1, cutoff=0.75
        )
        if matches:
            result[vn] = norm_to_torvik[matches[0]]
            continue

        # Step 5: more aggressive fuzzy on the stripped name
        stripped_lower = stripped.lower().replace(".", "").replace("'", "")
        matches = get_close_matches(
            stripped_lower, torvik_lower_list, n=1, cutoff=0.60
        )
        if matches:
            result[vn] = norm_to_torvik[matches[0]]

    return result


# ---------------------------------------------------------------------------
# Build rolling team stats from Torvik game log
# ---------------------------------------------------------------------------

def _team_game_rows(torvik: pd.DataFrame) -> pd.DataFrame:
    """Pivot Torvik game log into one-row-per-team-per-game with stats.

    Uses vectorized concat instead of row-by-row iteration.
    """
    base = ["date", "venue", "tempo", "possessions"]

    # Team 1 rows
    t1_src = base + [
        "team1", "team2",
        "t1adjt", "t1adjo", "t1adjd", "t1pts", "t2pts",
        "t1ppp", "t2ppp", "t1oe", "t1de",
        "t1rk", "t2rk", "t1qual", "t2qual",
    ]
    t1 = torvik[t1_src].copy()
    t1.columns = base + [
        "team", "opp",
        "adjt", "adjo", "adjd", "pts", "opp_pts",
        "ppp", "opp_ppp", "oe", "de",
        "rk", "opp_rk", "qual", "opp_qual",
    ]

    # Team 2 rows
    t2_src = base + [
        "team2", "team1",
        "t2adjt", "t2adjo", "t2adjd", "t2pts", "t1pts",
        "t2ppp", "t1ppp", "t2oe", "t2de",
        "t2rk", "t1rk", "t2qual", "t1qual",
    ]
    t2 = torvik[t2_src].copy()
    t2.columns = base + [
        "team", "opp",
        "adjt", "adjo", "adjd", "pts", "opp_pts",
        "ppp", "opp_ppp", "oe", "de",
        "rk", "opp_rk", "qual", "opp_qual",
    ]

    return pd.concat([t1, t2], ignore_index=True)


def build_rolling_stats(
    torvik: pd.DataFrame,
    windows: list[int] | None = None,
    ema_span: int = 10,
) -> pd.DataFrame:
    """
    For each team-game, compute rolling & EMA features *prior* to that game.

    This is a key improvement over KenPom: we capture recency-weighted
    trajectories, not just season-long adjusted averages.
    """
    if windows is None:
        windows = [5, 10, 20]

    tg = _team_game_rows(torvik)
    tg = tg.sort_values(["team", "date"]).reset_index(drop=True)

    stat_cols = ["adjt", "adjo", "adjd", "pts", "opp_pts", "ppp",
                 "opp_ppp", "tempo", "possessions"]

    grouped = tg.groupby("team")

    for col in stat_cols:
        # Shifted so we only use *prior* games (no lookahead)
        shifted = grouped[col].shift(1)
        # EMA
        tg[f"{col}_ema{ema_span}"] = (
            shifted.ewm(span=ema_span, min_periods=3).mean()
        )
        for w in windows:
            tg[f"{col}_roll{w}"] = (
                shifted.rolling(window=w, min_periods=max(2, w // 2)).mean()
            )
        # Season-to-date mean
        tg[f"{col}_season"] = shifted.expanding(min_periods=3).mean()

        # Volatility (std) – captures consistency
        tg[f"{col}_std5"] = (
            shifted.rolling(window=5, min_periods=3).std()
        )

    # Rolling opponent quality
    tg["opp_qual_ema"] = (
        grouped["opp_qual"].shift(1).ewm(span=ema_span, min_periods=3).mean()
    )
    tg["opp_rk_ema"] = (
        grouped["opp_rk"].shift(1).ewm(span=ema_span, min_periods=3).mean()
    )

    # Games played
    tg["games_played"] = grouped.cumcount()

    return tg


def get_latest_team_stats(rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Return the most recent rolling-stat row for every team.

    This gives us each team's latest EMA/rolling/season features — exactly
    what we need to build features for upcoming (un-played) games.
    """
    rolling = rolling.sort_values(["team", "date"])
    latest = rolling.groupby("team").tail(1).copy()
    latest = latest.set_index("team")
    return latest


def build_upcoming_rows(
    upcoming: pd.DataFrame,
    rolling: pd.DataFrame,
    name_map: dict,
) -> pd.DataFrame:
    """
    Build a model-ready DataFrame for upcoming games (no actual results).

    Parameters
    ----------
    upcoming : DataFrame with columns:
        home_team, away_team, total_point, spread_home_point
        (from odds_api.pick_consensus_line)
    rolling : DataFrame from build_rolling_stats
    name_map : dict mapping Odds API team names → Torvik team names

    Returns
    -------
    DataFrame with home/away rolling features, vegas_total, spread columns
    — ready for build_feature_matrix().
    """
    latest = get_latest_team_stats(rolling)

    rows = []
    for _, game in upcoming.iterrows():
        home_vegas = game["home_team"]
        away_vegas = game["away_team"]

        home_torvik = name_map.get(home_vegas)
        away_torvik = name_map.get(away_vegas)

        if home_torvik is None or away_torvik is None:
            continue
        if home_torvik not in latest.index or away_torvik not in latest.index:
            continue

        h_stats = latest.loc[home_torvik]
        a_stats = latest.loc[away_torvik]

        row = {}
        # Carry over game-level fields
        for col in ["game_id", "commence_time", "over_price", "under_price",
                     "n_books"]:
            if col in game.index:
                row[col] = game[col]

        row["home_team"] = home_vegas
        row["away_team"] = away_vegas
        row["home_torvik"] = home_torvik
        row["away_torvik"] = away_torvik
        row["closing_total_point"] = game["total_point"]
        row["vegas_total"] = game["total_point"]
        row["closing_spread_home_point"] = game.get("spread_home_point", 0) or 0

        # Use commence_time as the game_date for feature engineering
        ct = pd.to_datetime(game.get("commence_time"))
        row["game_date"] = ct if pd.notna(ct) else pd.Timestamp.now()

        # Attach home rolling stats with h_ prefix
        for col in h_stats.index:
            row[f"h_{col}"] = h_stats[col]
        # Attach away rolling stats with a_ prefix
        for col in a_stats.index:
            row[f"a_{col}"] = a_stats[col]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ensure game_date is datetime
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ---------------------------------------------------------------------------
# Merge Vegas + Torvik → feature matrix
# ---------------------------------------------------------------------------

def merge_datasets(
    vegas: pd.DataFrame,
    torvik_rolling: pd.DataFrame,
    name_map: dict,
) -> pd.DataFrame:
    """
    Join Vegas lines with Torvik rolling stats to produce the modelling
    dataset. Each row = one game with home/away features + Vegas line.
    """
    # Map Vegas names → Torvik names
    vegas = vegas.copy()
    vegas["home_torvik"] = vegas["home_team"].map(name_map)
    vegas["away_torvik"] = vegas["away_team"].map(name_map)

    # Drop rows where we couldn't map either team
    vegas = vegas.dropna(subset=["home_torvik", "away_torvik"])

    # Convert Vegas UTC dates to US Eastern for matching with Torvik
    # Torvik uses Eastern dates; Vegas game_datetime_utc is in UTC.
    # Subtract 5 hours (ET offset) to get the correct calendar date.
    vegas_dt = pd.to_datetime(vegas["game_datetime_utc"], utc=True, errors="coerce")
    vegas["game_date"] = (vegas_dt - pd.Timedelta(hours=5)).dt.tz_localize(None).dt.normalize()
    # Fallback for rows where datetime parsing failed
    fallback_mask = vegas["game_date"].isna()
    if fallback_mask.any():
        vegas.loc[fallback_mask, "game_date"] = pd.to_datetime(
            vegas.loc[fallback_mask, "game_date_utc"]
        ).dt.normalize()

    tr = torvik_rolling.copy()
    tr["game_date"] = pd.to_datetime(tr["date"]).dt.normalize()

    # Merge home stats
    home = tr.add_prefix("h_")
    merged = vegas.merge(
        home,
        left_on=["home_torvik", "game_date"],
        right_on=["h_team", "h_game_date"],
        how="inner",
    )

    # Merge away stats
    away = tr.add_prefix("a_")
    merged = merged.merge(
        away,
        left_on=["away_torvik", "game_date"],
        right_on=["a_team", "a_game_date"],
        how="inner",
    )

    # Actual total (sum of both teams' points from Torvik)
    merged["actual_total"] = merged["h_pts"] + merged["a_pts"]
    merged["vegas_total"] = merged["closing_total_point"]
    merged["abs_spread"] = merged["closing_spread_home_point"].abs()

    return merged
