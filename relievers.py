import pandas as pd
import numpy as np
from pybaseball import statcast, cache
from datetime import date, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

cache.enable()

SEASON        = 2026
LOOKBACK_DAYS = 14
MIN_BF        = 30

# ─── DATA PULL ────────────────────────────────────────────────────────────────

def get_season_statcast(season):
    start = f"{season}-03-27"
    end   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Pulling Statcast {start} → {end} ...")
    df = statcast(start_dt=start, end_dt=end)
    if df is None or df.empty:
        raise RuntimeError("No Statcast data returned.")
    print(f"  {len(df):,} pitches")
    return df

def classify_pitchers(df):
    inning1 = df[(df["inning"] == 1) & (df["outs_when_up"] == 0)]
    gs = (inning1.groupby(["pitcher","game_pk"]).size().reset_index()
          .groupby("pitcher").size().rename("GS"))
    g  = (df.groupby(["pitcher","game_pk"]).size().reset_index()
          .groupby("pitcher").size().rename("G"))
    info = pd.concat([gs, g], axis=1).fillna(0)
    info["gs_ratio"] = info["GS"] / info["G"]
    relievers = info[info["gs_ratio"] < 0.5].index
    starters  = info[info["gs_ratio"] >= 0.5].index
    return (df[df["pitcher"].isin(relievers)].copy(),
            df[df["pitcher"].isin(starters)].copy())

def get_pitcher_team(df):
    name_team = (df.sort_values("game_date")
                 .groupby("pitcher")[["player_name","home_team","away_team","inning_topbot"]]
                 .last().reset_index())
    name_team["team"] = np.where(
        name_team["inning_topbot"] == "Top",
        name_team["home_team"],
        name_team["away_team"]
    )
    return name_team[["pitcher","player_name","team"]]

# ─── RELIEVER STATS ───────────────────────────────────────────────────────────

def compute_reliever_stats(df):
    K_EV   = {"strikeout","strikeout_double_play"}
    BB_EV  = {"walk","hit_by_pitch"}
    H_EV   = {"single","double","triple","home_run"}
    OUT_EV = {"field_out","strikeout","force_out","grounded_into_double_play",
               "double_play","sac_fly","sac_bunt","fielders_choice_out",
               "strikeout_double_play","sac_fly_double_play","triple_play"}

    pa = df[df["events"].notna()].copy()
    pa["is_k"]   = pa["events"].isin(K_EV)
    pa["is_bb"]  = pa["events"].isin(BB_EV)
    pa["is_hit"] = pa["events"].isin(H_EV)
    pa["is_out"] = pa["events"].isin(OUT_EV)

    agg = pa.groupby("pitcher").agg(
        BF   =("events","count"),
        K    =("is_k","sum"),
        BB   =("is_bb","sum"),
        H    =("is_hit","sum"),
        outs =("is_out","sum"),
    ).reset_index()

    name_team = get_pitcher_team(df)
    agg = agg.merge(name_team, on="pitcher", how="left")

    g = (df.groupby(["pitcher","game_pk"]).size().reset_index()
         .groupby("pitcher").size().rename("G").reset_index())
    agg = agg.merge(g, on="pitcher", how="left")

    agg["IP"]   = (agg["outs"] / 3).round(2)
    agg["K%"]   = (agg["K"]  / agg["BF"] * 100).round(1)
    agg["BB%"]  = (agg["BB"] / agg["BF"] * 100).round(1)
    agg["KBB"]  = (agg["K"]  / agg["BB"].replace(0, np.nan)).round(2)
    agg["WHIP"] = ((agg["BB"] + agg["H"]) / agg["IP"].replace(0, np.nan)).round(2)

    return agg[agg["BF"] >= MIN_BF].rename(columns={"player_name":"name"})

def get_recent(df, lookback):
    today     = date.today()
    yesterday = today - timedelta(days=1)
    cutoff    = yesterday - timedelta(days=lookback - 1)

    df["gd"] = pd.to_datetime(df["game_date"]).dt.date
    recent   = df[(df["gd"] >= cutoff) & (df["gd"] <= yesterday)].copy()

    OUT_EV = {"field_out","strikeout","force_out","grounded_into_double_play",
              "double_play","sac_fly","sac_bunt","fielders_choice_out",
              "strikeout_double_play","sac_fly_double_play","triple_play"}

    apps = (recent.groupby(["pitcher","game_pk","gd"]).size().reset_index()
            [["pitcher","gd"]].drop_duplicates().sort_values(["pitcher","gd"]))

    pa = recent[recent["events"].notna()].copy()
    pa["is_out"] = pa["events"].isin(OUT_EV)
    outs = pa.groupby("pitcher")["is_out"].sum().rename("recent_outs").reset_index()

    rows = []
    for pid, grp in apps.groupby("pitcher"):
        dates    = sorted(grp["gd"].tolist())
        last     = dates[-1]
        rest     = (yesterday - last).days
        date_set = set(dates)
        consec   = 0
        check    = yesterday
        while True:
            if check in date_set:
                consec += 1
                check  -= timedelta(days=1)
            else:
                break
        rows.append({"pitcher":pid, "games_recent":len(dates),
                     "days_rested":rest, "consec":consec,
                     "app_dates":[str(d) for d in dates]})

    rdf = pd.DataFrame(rows).merge(outs, on="pitcher", how="left")
    rdf["ip_recent"] = (rdf["recent_outs"].fillna(0) / 3).round(2)
    return rdf

# ─── PITCH MIX ────────────────────────────────────────────────────────────────

PITCH_NAMES = {
    "FF":"4-Seam FB","SI":"Sinker",   "FC":"Cutter",
    "SL":"Slider",   "ST":"Sweeper",  "CU":"Curveball",
    "KC":"Knuckle-CB","CH":"Changeup","FS":"Splitter",
    "FO":"Forkball", "KN":"Knuckleball","EP":"Eephus",
    "CS":"Slow Curve","SC":"Screwball","SV":"Slurve",
    "FA":"Fastball",
}

PITCH_CATEGORY = {
    "FF":"FB","SI":"FB","FC":"FB","FA":"FB",
    "SL":"BR","ST":"BR","CU":"BR","KC":"BR","CS":"BR","SV":"BR",
    "CH":"OS","FS":"OS","FO":"OS","SC":"OS","KN":"OS","EP":"OS",
}

def compute_pitch_mix(df, role):
    valid = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
    if valid.empty:
        return pd.DataFrame()

    name_team      = get_pitcher_team(df)
    total_pitches  = valid.groupby("pitcher").size().rename("total")
    total_by_hand  = valid.groupby(["pitcher","stand"]).size().rename("total_hand").reset_index()

    agg = (valid.groupby(["pitcher","pitch_type","stand"])
           .agg(
               count    =("pitch_type","count"),
               avg_velo =("release_speed","mean"),
               max_velo =("release_speed","max"),
               avg_spin =("release_spin_rate","mean"),
           ).reset_index())

    agg = agg.merge(total_pitches,  on="pitcher")
    agg = agg.merge(total_by_hand,  on=["pitcher","stand"], how="left")

    agg["usage_pct"]      = (agg["count"] / agg["total"]      * 100).round(1)
    agg["usage_pct_hand"] = (agg["count"] / agg["total_hand"] * 100).round(1)
    agg["avg_velo"]       = agg["avg_velo"].round(1)
    agg["max_velo"]       = agg["max_velo"].round(1)
    agg["avg_spin"]       = agg["avg_spin"].round(0).astype("Int64")
    agg["pitch_name"]     = agg["pitch_type"].map(PITCH_NAMES).fillna(agg["pitch_type"])
    agg["category"]       = agg["pitch_type"].map(PITCH_CATEGORY).fillna("OS")
    agg["role"]           = role
    agg = agg.merge(name_team, on="pitcher", how="left")
    agg = agg.rename(columns={"player_name":"name"})
    agg = agg[agg["total"] >= 20]
    return agg

# ─── HTML ─────────────────────────────────────────────────────────────────────

def build_html(reliever_stats, reliever_recent, pitch_mix_rel, pitch_mix_sp, lookback, today):

    # ── reliever records ──
    rel_records = []
    for _, r in reliever_stats.iterrows():
        kpct = float(r["K%"]) if pd.notna(r["K%"]) else 0
        if kpct >= 35:   tier, tc = "ELITE", "#0F6E56"
        elif kpct >= 28: tier, tc = "GOOD",  "#3B6D11"
        elif kpct >= 20: tier, tc = "AVG",   "#BA7517"
        else:            tier, tc = "BELOW", "#A32D2D"

        consec = int(r.get("consec", 0))
        rest   = int(r.get("days_rested", 99))
        if consec >= 3:   status, sc = "FATIGUED",   "#A32D2D"
        elif consec == 2: status, sc = "USED",        "#BA7517"
        elif rest == 0:   status, sc = "USED TODAY",  "#BA7517"
        elif rest >= 3:   status, sc = "FRESH",       "#0F6E56"
        else:             status, sc = "NORMAL",      "#888780"

        dots = []
        date_set = set(r.get("app_dates", []))
        for i in range(lookback - 1, -1, -1):
            d = str(today - timedelta(days=i + 1))
            dots.append("1" if d in date_set else "0")

        rel_records.append({
            "name":         str(r["name"]),
            "team":         str(r["team"]),
            "kpct":         round(kpct, 1),
            "bbpct":        round(float(r["BB%"]) if pd.notna(r["BB%"]) else 0, 1),
            "kbb":          round(float(r["KBB"])  if pd.notna(r["KBB"])  else 0, 2),
            "whip":         round(float(r["WHIP"]) if pd.notna(r["WHIP"]) else 0, 2),
            "ip":           round(float(r["IP"])   if pd.notna(r["IP"])   else 0, 1),
            "g":            int(r["G"])  if pd.notna(r["G"])  else 0,
            "bf":           int(r["BF"]) if pd.notna(r["BF"]) else 0,
            "k":            int(r["K"])  if pd.notna(r["K"])  else 0,
            "bb":           int(r["BB"]) if pd.notna(r["BB"]) else 0,
            "games_recent": int(r.get("games_recent", 0)),
            "ip_recent":    round(float(r.get("ip_recent", 0)), 1),
            "days_rested":  rest,
            "consec":       consec,
            "tier":         tier,
            "tier_color":   tc,
            "status":       status,
            "status_color": sc,
            "dots":         "".join(dots),
        })

    # ── pitch mix records ──
    def pitch_records(pm_df):
        if pm_df.empty:
            return []
        rows = {}
        for _, r in pm_df.iterrows():
            pid   = int(r["pitcher"])
            stand = str(r["stand"])
            name  = str(r["name"])
            team  = str(r["team"])

            if pid not in rows:
                rows[pid] = {
                    "pitcher":    pid,
                    "name":       name,
                    "team":       team,
                    "total":      int(r["total"]),
                    "pitches_all":[],
                    "pitches_L":  [],
                    "pitches_R":  [],
                }

            entry = {
                "type":       str(r["pitch_type"]),
                "name":       str(r["pitch_name"]),
                "category":   str(r["category"]),
                "count":      int(r["count"]),
                "usage":      float(r["usage_pct"]),
                "usage_hand": float(r["usage_pct_hand"]),
                "avg_velo":   float(r["avg_velo"]) if pd.notna(r["avg_velo"]) else None,
                "max_velo":   float(r["max_velo"]) if pd.notna(r["max_velo"]) else None,
                "avg_spin":   int(r["avg_spin"])   if pd.notna(r["avg_spin"]) else None,
            }
            rows[pid][f"pitches_{stand}"].append(entry)

        for pid, d in rows.items():
            seen  = {}
            total = d["total"]
            for stand in ["L","R"]:
                for p in d[f"pitches_{stand}"]:
                    t = p["type"]
                    if t not in seen:
                        seen[t] = {**p, "count": 0}
                    seen[t]["count"] += p["count"]
            for t, p in seen.items():
                p["usage"] = round(p["count"] / total * 100, 1)
            d["pitches_all"] = sorted(seen.values(), key=lambda x: x["usage"], reverse=True)
            d["pitches_L"]   = sorted(d["pitches_L"], key=lambda x: x["usage_hand"], reverse=True)
            d["pitches_R"]   = sorted(d["pitches_R"], key=lambda x: x["usage_hand"], reverse=True)

            cat_totals = {"FB":0,"BR":0,"OS":0}
            cat_velo_weighted = {"FB":0,"BR":0,"OS":0}
            for p in d["pitches_all"]:
                cat = p["category"]
                cat_totals[cat] = cat_totals.get(cat, 0) + p["count"]
                if p["avg_velo"] is not None:
                    cat_velo_weighted[cat] += p["avg_velo"] * p["count"]
            d["cat_fb"] = round(cat_totals["FB"] / total * 100, 1) if total else 0
            d["cat_br"] = round(cat_totals["BR"] / total * 100, 1) if total else 0
            d["cat_os"] = round(cat_totals["OS"] / total * 100, 1) if total else 0
            d["cat_fb_velo"] = round(cat_velo_weighted["FB"] / cat_totals["FB"], 1) if cat_totals["FB"] > 0 else None
            d["cat_br_velo"] = round(cat_velo_weighted["BR"] / cat_totals["BR"], 1) if cat_totals["BR"] > 0 else None
            d["cat_os_velo"] = round(cat_velo_weighted["OS"] / cat_totals["OS"], 1) if cat_totals["OS"] > 0 else None

        return list(rows.values())

    rel_pitch = pitch_records(pitch_mix_rel)
    sp_pitch  = pitch_records(pitch_mix_sp)

    dot_labels = []
    for i in range(lookback - 1, -1, -1):
        d = today - timedelta(days=i + 1)
        dot_labels.append(f"{d.month}/{d.day}")

    rel_json          = json.dumps(rel_records)
    rel_pitch_json    = json.dumps(rel_pitch)
    sp_pitch_json     = json.dumps(sp_pitch)
    dot_labels_json   = json.dumps(dot_labels)
    pitch_colors_json = json.dumps({
        "FF":"#E05C5C","SI":"#E08A5C","FC":"#E0C45C","SL":"#8AE05C",
        "ST":"#5CE09E","CU":"#5CB8E0","KC":"#5C7AE0","CH":"#A05CE0",
        "FS":"#E05CA8","FO":"#A0A0A0","KN":"#E0E05C","CS":"#5CE0D8",
        "SC":"#C45CE0","SV":"#E07A5C","FA":"#E05C5C",
    })
    cat_colors_json = json.dumps({
        "FB":"#E05C5C","BR":"#5CB8E0","OS":"#A05CE0"
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLB Pitcher Dashboard {SEASON}</title>
<style>
  :root {{
    --bg:      #0e1117;
    --surface: #161b27;
    --card:    #1e2535;
    --border:  #2a3347;
    --text:    #e8eaf0;
    --muted:   #7a8499;
    --accent:  #378ADD;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; }}

  .tab-bar {{ display: flex; border-bottom: 1px solid var(--border); padding: 0 28px; margin-top: 8px; }}
  .tab-btn {{ padding: 12px 22px; font-size: 13px; font-weight: 500; cursor: pointer; border: none; background: none; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all .15s; }}
  .tab-btn:hover {{ color: var(--text); }}
  .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  .header {{ padding: 24px 28px 0; }}
  .header h1 {{ font-size: 22px; font-weight: 600; letter-spacing: -.3px; }}
  .header p  {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

  .summary {{ display: flex; gap: 12px; padding: 16px 28px; flex-wrap: wrap; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 18px; min-width: 110px; }}
  .card-label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .card-val {{ font-size: 22px; font-weight: 600; }}

  .controls {{ display: flex; gap: 10px; padding: 12px 28px; flex-wrap: wrap; align-items: center; border-bottom: 1px solid var(--border); }}
  .controls input, .controls select {{ background: var(--card); border: 1px solid var(--border); border-radius: 7px; color: var(--text); font-size: 12px; padding: 7px 11px; outline: none; }}
  .controls input {{ width: 200px; }}
  .controls input:focus, .controls select:focus {{ border-color: var(--accent); }}
  .controls label {{ font-size: 11px; color: var(--muted); }}
  .btn-group {{ display: flex; gap: 6px; margin-left: auto; }}
  .pill-btn {{ font-size: 11px; padding: 5px 12px; border-radius: 20px; cursor: pointer; border: 1px solid var(--border); background: var(--card); color: var(--muted); transition: all .15s; }}
  .pill-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  .table-wrap {{ overflow-x: auto; padding: 0 28px 28px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 14px; min-width: 900px; }}
  thead th {{ font-size: 11px; font-weight: 500; color: var(--muted); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; white-space: nowrap; position: sticky; top: 0; background: var(--bg); z-index: 2; }}
  thead th:hover {{ color: var(--text); }}
  thead th.sorted {{ color: var(--accent); }}
  thead th .arr {{ margin-left: 3px; opacity: .5; font-size: 10px; }}
  thead th.sorted .arr {{ opacity: 1; }}
  tbody tr {{ border-bottom: 1px solid var(--border); transition: background .1s; }}
  tbody tr:hover {{ background: var(--card); }}
  tbody td {{ padding: 9px 10px; vertical-align: middle; white-space: nowrap; }}
  .name-cell {{ font-weight: 500; }}
  .team-badge {{ display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 4px; background: var(--card); color: var(--muted); border: 1px solid var(--border); letter-spacing: .4px; }}
  .kpct-wrap {{ display: flex; align-items: center; gap: 7px; }}
  .kpct-bar-bg {{ width: 70px; height: 5px; border-radius: 3px; background: var(--border); flex-shrink: 0; }}
  .kpct-bar-fill {{ height: 5px; border-radius: 3px; }}
  .kpct-val {{ font-weight: 600; min-width: 38px; }}
  .tier-pill {{ font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: .5px; }}
  .status-pill {{ font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px; letter-spacing: .4px; }}
  .dots-wrap {{ display: flex; gap: 3px; align-items: center; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; cursor: default; position: relative; }}
  .dot:hover::after {{ content: attr(data-tip); position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%); background: #1e2535; border: 1px solid var(--border); border-radius: 5px; padding: 3px 7px; font-size: 10px; white-space: nowrap; color: var(--text); pointer-events: none; z-index: 10; }}
  .legend {{ display: flex; gap: 18px; padding: 0 28px; flex-wrap: wrap; margin-bottom: 4px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--muted); }}
  .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; }}
  .row-count {{ font-size: 11px; color: var(--muted); padding: 6px 28px 0; }}

  .pitch-controls {{ display: flex; gap: 10px; padding: 16px 28px 12px; flex-wrap: wrap; align-items: center; border-bottom: 1px solid var(--border); }}
  .pitch-controls input, .pitch-controls select {{ background: var(--card); border: 1px solid var(--border); border-radius: 7px; color: var(--text); font-size: 12px; padding: 7px 11px; outline: none; }}
  .pitch-controls input {{ width: 180px; }}
  .pitch-controls input:focus, .pitch-controls select:focus {{ border-color: var(--accent); }}
  .pitch-controls label {{ font-size: 11px; color: var(--muted); }}

  .toggle-group {{ display: flex; background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  .toggle-btn {{ padding: 6px 14px; font-size: 12px; cursor: pointer; border: none; background: none; color: var(--muted); transition: all .15s; white-space: nowrap; }}
  .toggle-btn.active {{ background: var(--accent); color: #fff; }}
  .toggle-btn.active.fb {{ background: #E05C5C; }}
  .toggle-btn.active.br {{ background: #5CB8E0; color: #000; }}
  .toggle-btn.active.os {{ background: #A05CE0; }}

  .pitch-grid {{ padding: 16px 28px; display: grid; grid-template-columns: repeat(auto-fill, minmax(360px,1fr)); gap: 14px; }}
  .pitcher-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .pitcher-card-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }}
  .pitcher-card-left {{ display: flex; flex-direction: column; gap: 5px; }}
  .pitcher-card-name {{ font-weight: 600; font-size: 13px; }}
  .cat-badges {{ display: flex; gap: 5px; flex-wrap: wrap; }}
  .cat-badge {{ font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: .3px; }}
  .pitcher-card-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }}
  .pitcher-card-meta {{ font-size: 11px; color: var(--muted); }}
  .pitch-row {{ margin-bottom: 8px; }}
  .pitch-row-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }}
  .pitch-label {{ font-size: 12px; font-weight: 500; }}
  .pitch-stats {{ font-size: 11px; color: var(--muted); display: flex; gap: 8px; align-items: center; }}
  .pitch-stat-val {{ color: var(--text); font-weight: 500; }}
  .pitch-bar-bg {{ width: 100%; height: 5px; border-radius: 3px; background: var(--border); }}
  .pitch-bar-fill {{ height: 5px; border-radius: 3px; }}
  .no-results {{ text-align: center; padding: 3rem; color: var(--muted); font-size: 13px; grid-column: 1/-1; }}
  .pitch-count-label {{ font-size: 11px; color: var(--muted); padding: 8px 28px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>⚾ MLB Pitcher Dashboard — {SEASON}</h1>
  <p>Statcast-derived stats · Min {MIN_BF} BF · Updated {today}</p>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('relievers',this)">Reliever K% &amp; Usage</button>
  <button class="tab-btn"        onclick="switchTab('pitchmix', this)">Pitch Speed &amp; Mix</button>
</div>

<!-- ══════════════ TAB 1 ══════════════ -->
<div id="tab-relievers" class="tab-content active">
  <div class="summary" id="summaryCards"></div>
  <div class="controls">
    <input type="text" id="search" placeholder="Search name or team..." oninput="renderRelievers()" />
    <div><label>Team&nbsp;</label>
      <select id="teamSel" onchange="renderRelievers()"><option value="">All</option></select></div>
    <div><label>Min IP&nbsp;</label>
      <select id="minIP" onchange="renderRelievers()">
        <option value="0">Any</option><option value="5">5+</option>
        <option value="10">10+</option><option value="20">20+</option>
      </select></div>
    <div><label>Recent&nbsp;</label>
      <select id="recentSel" onchange="renderRelievers()">
        <option value="0">Any</option><option value="1">Pitched today</option>
        <option value="2">Last 2 days</option><option value="3">Last 3 days</option>
      </select></div>
    <div class="btn-group">
      <button class="pill-btn active" data-tier=""      onclick="setTier(this)">All</button>
      <button class="pill-btn"        data-tier="ELITE" onclick="setTier(this)">Elite</button>
      <button class="pill-btn"        data-tier="GOOD"  onclick="setTier(this)">Good</button>
      <button class="pill-btn"        data-tier="AVG"   onclick="setTier(this)">Avg</button>
      <button class="pill-btn"        data-tier="BELOW" onclick="setTier(this)">Below</button>
    </div>
  </div>
  <div class="legend" style="margin-top:12px;">
    <span class="legend-item"><span class="legend-dot" style="background:#0F6E56"></span>Elite (35%+)</span>
    <span class="legend-item"><span class="legend-dot" style="background:#3B6D11"></span>Good (28–35%)</span>
    <span class="legend-item"><span class="legend-dot" style="background:#BA7517"></span>Avg (20–28%)</span>
    <span class="legend-item"><span class="legend-dot" style="background:#A32D2D"></span>Below (&lt;20%)</span>
    <span style="margin-left:12px" class="legend-item"><span class="legend-dot" style="background:#378ADD"></span>Pitched</span>
    <span class="legend-item"><span class="legend-dot" style="background:#2a3347;border:1px solid #4a5568"></span>Rest</span>
    <span class="legend-item"><span class="legend-dot" style="background:#A32D2D"></span>3+ consec</span>
  </div>
  <div class="row-count" id="rowCount"></div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th data-col="name"         data-dir="-1">Player <span class="arr">↕</span></th>
        <th data-col="team"         data-dir="-1">Team <span class="arr">↕</span></th>
        <th data-col="kpct"         data-dir="-1" class="sorted">K% <span class="arr">↓</span></th>
        <th data-col="tier"         data-dir="-1">Tier <span class="arr">↕</span></th>
        <th data-col="bbpct"        data-dir="-1">BB% <span class="arr">↕</span></th>
        <th data-col="kbb"          data-dir="-1">K/BB <span class="arr">↕</span></th>
        <th data-col="whip"         data-dir="-1">WHIP <span class="arr">↕</span></th>
        <th data-col="ip"           data-dir="-1">IP <span class="arr">↕</span></th>
        <th data-col="g"            data-dir="-1">G <span class="arr">↕</span></th>
        <th data-col="games_recent" data-dir="-1">G-{lookback}d <span class="arr">↕</span></th>
        <th data-col="days_rested"  data-dir="-1">Rest <span class="arr">↕</span></th>
        <th data-col="status"       data-dir="-1">Status <span class="arr">↕</span></th>
        <th>Last {lookback} days</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<!-- ══════════════ TAB 2 ══════════════ -->
<div id="tab-pitchmix" class="tab-content">
  <div class="pitch-controls">
    <input type="text" id="pmSearch" placeholder="Search name or team..." oninput="renderPitchMix()" />
    <div><label>Team&nbsp;</label>
      <select id="pmTeam" onchange="renderPitchMix()"><option value="">All teams</option></select></div>
    <div><label>Pitch type&nbsp;</label>
      <select id="pmPitchType" onchange="renderPitchMix()"><option value="">All pitches</option></select></div>
    <div><label>Min pitches&nbsp;</label>
      <select id="pmMinPitches" onchange="renderPitchMix()">
        <option value="20">20+</option>
        <option value="50">50+</option>
        <option value="100">100+</option>
        <option value="200">200+</option>
        <option value="300">300+</option>
      </select></div>
    <div class="toggle-group">
      <button class="toggle-btn active" data-hand="all" onclick="setHand(this)">All batters</button>
      <button class="toggle-btn"        data-hand="R"   onclick="setHand(this)">vs RHB</button>
      <button class="toggle-btn"        data-hand="L"   onclick="setHand(this)">vs LHB</button>
    </div>
    <div class="toggle-group">
      <button class="toggle-btn active" data-cat=""   onclick="setCat(this)">All types</button>
      <button class="toggle-btn fb"     data-cat="FB" onclick="setCat(this)">Fastball</button>
      <button class="toggle-btn br"     data-cat="BR" onclick="setCat(this)">Breaking</button>
      <button class="toggle-btn os"     data-cat="OS" onclick="setCat(this)">Off-Speed</button>
    </div>
    <div class="toggle-group">
      <button class="toggle-btn active" data-role="relievers" onclick="setRole(this)">Relievers</button>
      <button class="toggle-btn"        data-role="starters"  onclick="setRole(this)">Starters</button>
    </div>
  </div>
  <div class="pitch-count-label" id="pmCount"></div>
  <div class="pitch-grid" id="pitchGrid"></div>
</div>

<script>
const REL_DATA        = {rel_json};
const REL_PITCH_DATA  = {rel_pitch_json};
const SP_PITCH_DATA   = {sp_pitch_json};
const DOT_LABELS      = {dot_labels_json};
const PITCH_COLORS    = {pitch_colors_json};
const CAT_COLORS      = {cat_colors_json};
const LOOKBACK        = {lookback};

// ── UTILITIES ─────────────────────────────────────────────────────────────
function normalizeString(str) {{
  return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}}

// ── TAB SWITCHING ─────────────────────────────────────────────────────────
function switchTab(name, btn) {{
  document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  btn.classList.add("active");
}}

// ── TAB 1: RELIEVERS ──────────────────────────────────────────────────────
let sortCol = "kpct", sortDir = 1, tierFilter = "";

function setTier(btn) {{
  document.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  tierFilter = btn.dataset.tier;
  renderRelievers();
}}

const teams = [...new Set(REL_DATA.map(d => d.team))].sort();
const teamSel = document.getElementById("teamSel");
teams.forEach(t => {{ const o = document.createElement("option"); o.value=t; o.textContent=t; teamSel.appendChild(o); }});

function filteredRel() {{
  const q      = normalizeString(document.getElementById("search").value.toLowerCase());
  const team   = teamSel.value;
  const minIP  = parseFloat(document.getElementById("minIP").value);
  const recent = parseInt(document.getElementById("recentSel").value);
  return REL_DATA.filter(d => {{
    if (q && !normalizeString(d.name.toLowerCase()).includes(q) && !normalizeString(d.team.toLowerCase()).includes(q)) return false;
    if (team && d.team !== team) return false;
    if (d.ip < minIP) return false;
    if (tierFilter && d.tier !== tierFilter) return false;
    if (recent > 0 && d.days_rested >= recent) return false;
    return true;
  }});
}}

function sortedRel(arr) {{
  return [...arr].sort((a,b) => {{
    let av = a[sortCol], bv = b[sortCol];
    if (typeof av === "string") return sortDir * av.localeCompare(bv);
    return sortDir * (bv - av);
  }});
}}

function dotHtml(dots) {{
  let consec = 0;
  for (let i = dots.length-1; i >= 0; i--) {{ if (dots[i]==="1") consec++; else break; }}
  return dots.split("").map((v,i) => {{
    const label  = DOT_LABELS[i] || "";
    const isLast = (dots.length-1-i) < consec;
    const bg     = v==="0" ? "#2a3347" : (isLast && consec>=3 ? "#A32D2D" : "#378ADD");
    return `<span class="dot" style="background:${{bg}}" data-tip="${{label}}"></span>`;
  }}).join("");
}}

function kpctBar(val, color) {{
  const w = Math.min(val/45*100,100).toFixed(1);
  return `<div class="kpct-wrap">
    <div class="kpct-bar-bg"><div class="kpct-bar-fill" style="width:${{w}}%;background:${{color}}"></div></div>
    <span class="kpct-val" style="color:${{color}}">${{val.toFixed(1)}}%</span>
  </div>`;
}}

function renderRelievers() {{
  const rows = sortedRel(filteredRel());
  const avgK = rows.length ? (rows.reduce((s,r)=>s+r.kpct,0)/rows.length).toFixed(1) : "—";
  document.getElementById("summaryCards").innerHTML = `
    <div class="card"><div class="card-label">Avg K%</div><div class="card-val">${{avgK}}%</div></div>
    <div class="card"><div class="card-label">Fatigued</div><div class="card-val" style="color:#A32D2D">${{rows.filter(r=>r.consec>=3).length}}</div></div>`;
  document.getElementById("rowCount").textContent = `Showing ${{rows.length}} relievers`;
  const tbody = document.getElementById("tbody");
  if (!rows.length) {{
    tbody.innerHTML = `<tr><td colspan="13" style="text-align:center;padding:2rem;color:var(--muted)">No results</td></tr>`;
    return;
  }}
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td class="name-cell">${{r.name}}</td>
      <td><span class="team-badge">${{r.team}}</span></td>
      <td>${{kpctBar(r.kpct, r.tier_color)}}</td>
      <td><span class="tier-pill" style="background:${{r.tier_color}}22;color:${{r.tier_color}}">${{r.tier}}</span></td>
      <td style="color:var(--muted)">${{r.bbpct.toFixed(1)}}%</td>
      <td>${{r.kbb>0?r.kbb.toFixed(2):"—"}}</td>
      <td>${{r.whip>0?r.whip.toFixed(2):"—"}}</td>
      <td>${{r.ip.toFixed(1)}}</td>
      <td>${{r.g}}</td>
      <td>${{r.games_recent}}</td>
      <td style="color:${{r.days_rested<=1?"#BA7517":r.days_rested>=3?"#0F6E56":"var(--text)"}}">${{r.days_rested===99?"—":r.days_rested+"d"}}</td>
      <td><span class="status-pill" style="background:${{r.status_color}}22;color:${{r.status_color}}">${{r.status}}</span></td>
      <td><div class="dots-wrap">${{dotHtml(r.dots)}}</div></td>
    </tr>`).join("");
}}

document.querySelectorAll("thead th[data-col]").forEach(th => {{
  th.addEventListener("click", () => {{
    const col = th.dataset.col;
    sortDir = col === sortCol ? sortDir * -1 : -1;
    sortCol = col;
    document.querySelectorAll("thead th").forEach(h => {{
      h.classList.remove("sorted");
      const a = h.querySelector(".arr"); if (a) a.textContent = "↕";
    }});
    th.classList.add("sorted");
    const a = th.querySelector(".arr"); if (a) a.textContent = sortDir===-1?"↓":"↑";
    renderRelievers();
  }});
}});

// ── TAB 2: PITCH MIX ──────────────────────────────────────────────────────
let currentRole = "relievers";
let currentHand = "all";
let currentCat  = "";

function setRole(btn) {{
  document.querySelectorAll("#roleToggle .toggle-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentRole = btn.dataset.role;
  populatePmFilters();
  renderPitchMix();
}}

function setHand(btn) {{
  document.querySelectorAll("#handToggle .toggle-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentHand = btn.dataset.hand;
  renderPitchMix();
}}

function setCat(btn) {{
  document.querySelectorAll("#catToggle .toggle-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentCat = btn.dataset.cat;
  renderPitchMix();
}}

function currentPitchData() {{
  return currentRole === "relievers" ? REL_PITCH_DATA : SP_PITCH_DATA;
}}

function populatePmFilters() {{
  const data = currentPitchData();
  const pmTeam = document.getElementById("pmTeam");
  const allTeams = [...new Set(data.map(d => d.team))].sort();
  pmTeam.innerHTML = '<option value="">All teams</option>' +
    allTeams.map(t => `<option value="${{t}}">${{t}}</option>`).join("");

  const pmPitchType = document.getElementById("pmPitchType");
  const allTypes = [...new Set(data.flatMap(d => d.pitches_all.map(p => p.type)))].sort();
  pmPitchType.innerHTML = '<option value="">All pitches</option>' +
    allTypes.map(t => `<option value="${{t}}">${{t}}</option>`).join("");
}}

function getPitchList(pitcher) {{
  const key = currentHand === "all" ? "pitches_all"
            : currentHand === "L"   ? "pitches_L"
            :                         "pitches_R";
  let pitches = pitcher[key] || [];
  if (currentCat) pitches = pitches.filter(p => p.category === currentCat);
  return pitches;
}}

function renderPitchMix() {{
  const q         = normalizeString(document.getElementById("pmSearch").value.toLowerCase());
  const team      = document.getElementById("pmTeam").value;
  const pitchType = document.getElementById("pmPitchType").value;
  const minP      = parseInt(document.getElementById("pmMinPitches").value);
  const data      = currentPitchData();

  let filtered = data.filter(d => {{
    if (d.total < minP) return false;
    if (q && !normalizeString(d.name.toLowerCase()).includes(q) && !normalizeString(d.team.toLowerCase()).includes(q)) return false;
    if (team && d.team !== team) return false;
    if (pitchType && !d.pitches_all.some(p => p.type === pitchType)) return false;
    if (currentCat && !d.pitches_all.some(p => p.category === currentCat)) return false;
    return true;
  }});

  filtered = [...filtered].sort((a,b) => a.name.localeCompare(b.name));

  document.getElementById("pmCount").textContent =
    `Showing ${{filtered.length}} ${{currentRole}} · ${{filtered.reduce((s,d)=>s+d.total,0).toLocaleString()}} total pitches`;

  const grid = document.getElementById("pitchGrid");
  if (!filtered.length) {{
    grid.innerHTML = `<div class="no-results">No pitchers match the current filters.</div>`;
    return;
  }}

  grid.innerHTML = filtered.map(pitcher => {{
    let pitches = getPitchList(pitcher);
    if (pitchType) pitches = pitches.filter(p => p.type === pitchType);

    // calculate category stats based on filtered pitches
    const catStats = {{"FB": {{count: 0, velo: 0}}, "BR": {{count: 0, velo: 0}}, "OS": {{count: 0, velo: 0}}}};
    const totalPitches = pitches.reduce((s, p) => s + p.count, 0);
    pitches.forEach(p => {{
      const cat = p.category;
      catStats[cat].count += p.count;
      if (p.avg_velo != null) catStats[cat].velo += p.avg_velo * p.count;
    }});

    // category badges
    const catBadges = [];
    ["FB", "BR", "OS"].forEach(cat => {{
      const stat = catStats[cat];
      if (stat.count > 0) {{
        const pct = (stat.count / totalPitches * 100).toFixed(1);
        const vel = (stat.velo / stat.count).toFixed(1);
        const color = {{"FB":"#E05C5C", "BR":"#5CB8E0", "OS":"#A05CE0"}}[cat];
        const label = {{"FB":"FB", "BR":"BR", "OS":"OS"}}[cat];
        catBadges.push(`<span class="cat-badge" style="background:${{color}}22;color:${{color}}">${{label}} ${{pct}}% · ${{vel}} mph</span>`);
      }}
    }});

    const usageKey = currentHand === "all" ? "usage" : "usage_hand";

    const pitchRows = pitches.map(p => {{
      const color = PITCH_COLORS[p.type] || "#888780";
      const velo  = p.avg_velo != null ? `<span class="pitch-stat-val">${{p.avg_velo}}</span> mph` : "—";
      const maxV  = p.max_velo != null ? `max <span class="pitch-stat-val">${{p.max_velo}}</span>` : "";
      const spin  = p.avg_spin != null ? `<span class="pitch-stat-val">${{p.avg_spin.toLocaleString()}}</span> rpm` : "—";
      const usage = p[usageKey] != null ? p[usageKey].toFixed(1) : p.usage.toFixed(1);
      const barW  = Math.min(parseFloat(usage), 100).toFixed(1);
      return `
        <div class="pitch-row">
          <div class="pitch-row-top">
            <span class="pitch-label" style="color:${{color}}">${{p.name}}</span>
            <div class="pitch-stats">
              ${{velo}} ${{maxV ? "· " + maxV : ""}}
              · ${{spin}}
              · <span class="pitch-stat-val">${{usage}}%</span>
            </div>
          </div>
          <div class="pitch-bar-bg">
            <div class="pitch-bar-fill" style="width:${{barW}}%;background:${{color}}"></div>
          </div>
        </div>`;
    }}).join("");

    const handNote = currentHand !== "all"
      ? `<div style="font-size:10px;color:var(--muted);margin-top:6px;padding-top:6px;border-top:1px solid var(--border);text-align:right">Usage % vs ${{currentHand === "L" ? "LHB" : "RHB"}} only</div>`
      : "";

    return `
      <div class="pitcher-card">
        <div class="pitcher-card-header">
          <div class="pitcher-card-left">
            <span class="pitcher-card-name">${{pitcher.name}}</span>
            <div class="cat-badges">${{catBadges.join("")}}</div>
          </div>
          <div class="pitcher-card-right">
            <span class="team-badge">${{pitcher.team}}</span>
            <span class="pitcher-card-meta">${{pitcher.total.toLocaleString()}} pitches</span>
          </div>
        </div>
        ${{pitchRows || '<div style="font-size:11px;color:var(--muted);text-align:center;padding:8px 0">No pitches match filter</div>'}}
        ${{handNote}}
      </div>`;
  }}).join("");
}}

// ── INIT ──────────────────────────────────────────────────────────────────
// wire up toggle group IDs (needed for setHand/setCat/setRole selectors)
document.querySelector("[data-hand='all']").closest(".toggle-group").id = "handToggle";
document.querySelector("[data-cat='']").closest(".toggle-group").id     = "catToggle";
document.querySelector("[data-role='relievers']").closest(".toggle-group").id = "roleToggle";

renderRelievers();
populatePmFilters();
renderPitchMix();
</script>
</body>
</html>"""
    return html

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    today = date.today()

    raw = get_season_statcast(SEASON)

    print("Classifying pitchers...")
    rel_pitches, sp_pitches = classify_pitchers(raw)
    print(f"  Reliever pitches: {len(rel_pitches):,}  |  Starter pitches: {len(sp_pitches):,}")

    print("Computing reliever stats...")
    stats = compute_reliever_stats(rel_pitches)
    print(f"  Relievers with {MIN_BF}+ BF: {len(stats)}")

    print(f"Computing last {LOOKBACK_DAYS}-day appearances...")
    recent = get_recent(rel_pitches, LOOKBACK_DAYS)

    final = stats.merge(recent, on="pitcher", how="left")
    final["games_recent"] = final["games_recent"].fillna(0).astype(int)
    final["days_rested"]  = final["days_rested"].fillna(99).astype(int)
    final["consec"]       = final["consec"].fillna(0).astype(int)
    final["ip_recent"]    = final["ip_recent"].fillna(0.0)
    final["app_dates"]    = final["app_dates"].apply(lambda x: x if isinstance(x, list) else [])

    print("Computing pitch mix...")
    pitch_rel = compute_pitch_mix(rel_pitches, "reliever")
    pitch_sp  = compute_pitch_mix(sp_pitches,  "starter")
    print(f"  Relievers: {pitch_rel['pitcher'].nunique() if not pitch_rel.empty else 0}  |  Starters: {pitch_sp['pitcher'].nunique() if not pitch_sp.empty else 0}")

    html = build_html(final, recent, pitch_rel, pitch_sp, LOOKBACK_DAYS, today)

    out = "index.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Saved → {out}")
    print(f"  Open index.html in any browser")