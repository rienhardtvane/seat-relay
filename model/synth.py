"""Synthetic season-ticket data for the Seat Relay model.

Everything here is invented. FC Bayern publishes no no-show rate, season-ticket
count or exchange volume, so the generator encodes plausible behaviour drawn
from published research on season-ticket no-shows (habit, kick-off slot,
fixture appeal, travel distance) rather than real club data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS = {
    "Sat 15:30": 0.0,
    "Sat 18:30": 0.15,
    "Sun 17:30": 0.25,
    "Fri 20:30": 0.45,
    "Tue/Wed 21:00": 0.90,
}
COMPETITIONS = {"Bundesliga": 0.0, "Champions League": -0.70, "DFB-Pokal": 0.20}
DOMICILE = {"Munich": 0.0, "Bavaria": 0.20, "Rest of Germany": 0.55, "Abroad": 0.90}


def make_fixtures(season: int, rng: np.random.Generator, n: int = 22) -> pd.DataFrame:
    """One season of home fixtures with slot, competition and opponent appeal."""
    comp = rng.choice(list(COMPETITIONS), size=n, p=[0.77, 0.18, 0.05])
    slot = []
    for c in comp:
        if c == "Bundesliga":
            slot.append(rng.choice(list(SLOTS)[:4], p=[0.45, 0.25, 0.20, 0.10]))
        else:
            slot.append("Tue/Wed 21:00")
    appeal = rng.uniform(0, 1, size=n)  # 1 = marquee opponent
    return pd.DataFrame(
        {
            "season": season,
            "fixture_id": [f"S{season}-F{i + 1:02d}" for i in range(n)],
            "match_no": np.arange(1, n + 1),
            "competition": comp,
            "slot": slot,
            "opponent_appeal": appeal.round(2),
        }
    )


def make_holders(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Season-ticket holders with a latent habit and a midweek sensitivity."""
    return pd.DataFrame(
        {
            "holder_id": [f"H{i:05d}" for i in range(n)],
            "domicile": rng.choice(list(DOMICILE), size=n, p=[0.45, 0.35, 0.15, 0.05]),
            "age": rng.integers(18, 80, size=n),
            "tenure_years": rng.integers(1, 30, size=n),
            # latent propensity to skip; most holders rarely miss, a tail misses often
            "habit": rng.normal(-2.6, 1.0, size=n),
            # some holders find late midweek games much harder than others
            "midweek_sensitivity": rng.gamma(2.0, 0.6, size=n),
        }
    )


def simulate_attendance(holders: pd.DataFrame, fixtures: pd.DataFrame, rng) -> pd.DataFrame:
    """Cross holders with fixtures and draw a no-show outcome per seat."""
    df = holders.merge(fixtures, how="cross")
    midweek = (df["slot"] == "Tue/Wed 21:00").astype(float)
    logit = (
        df["habit"]
        + df["slot"].map(SLOTS) * 0.6
        + midweek * df["midweek_sensitivity"] * 0.8
        + df["competition"].map(COMPETITIONS)
        - df["opponent_appeal"] * 0.9
        + df["domicile"].map(DOMICILE)
        - (df["tenure_years"] > 15) * 0.25
        + (df["age"] < 25) * 0.20
    )
    p = 1 / (1 + np.exp(-logit))
    df["no_show"] = (rng.uniform(size=len(df)) < p).astype(int)
    return df.drop(columns=["habit", "midweek_sensitivity"])


def build(n_holders: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Two seasons of scans: season 1 for training, season 2 for testing."""
    rng = np.random.default_rng(seed)
    holders = make_holders(n_holders, rng)
    seasons = [simulate_attendance(holders, make_fixtures(s, rng), rng) for s in (1, 2)]
    return pd.concat(seasons, ignore_index=True).sort_values(
        ["holder_id", "season", "match_no"], ignore_index=True
    )


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features a club can compute before kick-off from the holder's own scans only."""
    df = df.copy()
    g = df.groupby("holder_id", sort=False)
    attended = 1 - df["no_show"]
    # matches attended in the previous 13 home games (the window used in the
    # Bundesliga field study), shifted so a match never sees its own outcome
    df["attended_last13"] = g["no_show"].transform(
        lambda s: (1 - s).shift(1).rolling(13, min_periods=1).sum()
    ).fillna(0)
    df["games_seen"] = g.cumcount()
    df["attend_rate_prior"] = (
        attended.groupby(df["holder_id"]).transform(lambda s: s.shift(1).expanding().mean())
    ).fillna(0.85)
    is_mid = df["slot"] == "Tue/Wed 21:00"
    mid_att = attended.where(is_mid)
    df["midweek_missed_prior"] = (
        df["no_show"].where(is_mid, 0).groupby(df["holder_id"]).transform(lambda s: s.shift(1).cumsum())
    ).fillna(0)
    df["midweek_seen_prior"] = (
        is_mid.astype(int).groupby(df["holder_id"]).transform(lambda s: s.shift(1).cumsum())
    ).fillna(0)
    df["midweek_attend_rate_prior"] = np.where(
        df["midweek_seen_prior"] > 0,
        1 - df["midweek_missed_prior"] / df["midweek_seen_prior"].clip(lower=1),
        df["attend_rate_prior"],
    )
    # consecutive misses of the same slot, used for the plain-language reason
    df["same_slot_missed_streak"] = 0
    streaks = []
    for _, h in df.groupby("holder_id", sort=False):
        last = {}
        out = []
        for slot, ns in zip(h["slot"], h["no_show"]):
            out.append(last.get(slot, 0))
            last[slot] = last.get(slot, 0) + 1 if ns else 0
        streaks.extend(out)
    df["same_slot_missed_streak"] = streaks
    del mid_att
    return df
