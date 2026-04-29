import pandas as pd
import numpy as np
from pybaseball import statcast, cache
from datetime import date, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

cache.enable()

SEASON        = 2025
LOOKBACK_DAYS = 14
MIN_BF        = 30

# ─── DATA PULL ────────────────────────────────────────────────────────────────

def get_season_statcast(season):
    start = f"{season}-03-20"
    end   = date.today().strftime("%Y-%m-%d")
    print(f"Pulling Statcast {start} → {end} ...")
    df = statcast(start_dt=start, end_dt=end)
    if df is None or df.empty:
        raise RuntimeError("No Statcast data returned.")
    print(f"  {len(df):,} pitches")
    return df

def classify_relievers(df):
    inning1 = df[(df["inning"] == 1) & (df["outs_when_up"] == 0)]
    gs = (inning1.groupby(["pitcher","game_pk"]).size().reset_index()
          .groupby("pitcher").size().rename("GS"))
    g  = (df.groupby(["pitcher","game_pk"]).size().reset_index()
          .groupby("pitcher").size().rename("G"))
    info = pd.concat([gs, g], axis=1).fillna(0)
    info["gs_ratio"] = info["GS"] / info["G"]
    relievers = info[info["gs_ratio"] < 0.5].index
    return df[df["pitcher"].isin(relievers)].copy()

def compute_stats(df):
    K_EV  = {"strikeout","strikeout_double_play"}
    BB_EV = {"walk","hit_by_pitch"}
    H_EV  = {"single","double","triple","home_run"}
    OUT_EV= {"field_out","strikeout","force_out","grounded_into_double_play",
              "double_play","sac_fly","sac_bunt","fielders_choice_out",
              "strikeout_double_play","sac_fly_double_play","triple_play"}

    pa = df[df["events"].notna()].copy()
    pa["is_k"]   = pa["events"].isin(K_EV)
    pa["is_bb"]  = pa["events"].isin(BB_EV)
    pa["is_hit"] = pa["events"].isin(H_EV)
    pa["is_out"] = pa["events"].isin(OUT_EV)
    pa["is_hr"]  = pa["events"] == "home_run"

    agg = pa.groupby("pitcher").agg(
        BF   =("events","count"),
        K    =("is_k","sum"),
        BB   =("is_bb","sum"),
        H    =("is_hit","sum"),
        outs =("is_out","sum"),
        HR   =("is_hr","sum"),
    ).reset_index()

    # pitcher name + team (from last appearance)
    name_team = (df.sort_values("game_date")
                 .groupby("pitcher")[["player_name","home_team","away_team","inning_topbot"]]
                 .last().reset_index())
    name_team["team"] = np.where(
    name_team["inning_topbot"] == "Top",
    name_team["home_team"],   # home team pitches in top (batting visitors)
    name_team["away_team"]    # away team pitches in bottom (batting home)

)
    agg = agg.merge(name_team[["pitcher","player_name","team"]], on="pitcher", how="left")

    # games
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
    today = date.today()
    yesterday = today - timedelta(days=1)                    # ← exclude today
    cutoff = yesterday - timedelta(days=lookback - 1)        # ← window ends yesterday

    df["gd"] = pd.to_datetime(df["game_date"]).dt.date
    recent = df[(df["gd"] >= cutoff) & (df["gd"] <= yesterday)].copy()  # ← cap at yesterday

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
        rest     = (yesterday - last).days                   # ← days since last app, from yesterday
        date_set = set(dates)
        consec   = 0
        check    = yesterday                                 # ← start consecutive count from yesterday
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

# ─── HTML GENERATION ──────────────────────────────────────────────────────────

def build_html(df, lookback, today):
    records = []
    for _, r in df.iterrows():
        kpct = float(r["K%"]) if pd.notna(r["K%"]) else 0
        if kpct >= 35:   tier, tier_color = "ELITE",  "#0F6E56"
        elif kpct >= 28: tier, tier_color = "GOOD",   "#3B6D11"
        elif kpct >= 20: tier, tier_color = "AVG",    "#BA7517"
        else:            tier, tier_color = "BELOW",  "#A32D2D"

        consec = int(r.get("consec", 0))
        rest   = int(r.get("days_rested", 99))
        if consec >= 3:        status, sc = "FATIGUED",   "#A32D2D"
        elif consec == 2:      status, sc = "USED B2B",       "#BA7517"
        elif rest == 0:        status, sc = "USED YESTERDAY", "#BA7517"
        elif rest >= 3:        status, sc = "FRESH",      "#0F6E56"
        else:                  status, sc = "NORMAL",     "#888780"

        # build last-14-day dot string
        dots = []
        date_set = set(r.get("app_dates", []))
        for i in range(lookback - 1, -1, -1):
         d = str(today - timedelta(days=i + 1))                   # ← shift back one day
         dots.append("1" if d in date_set else "0")

        records.append({
            "name":         str(r["name"]),
            "team":         str(r["team"]),
            "kpct":         round(kpct, 1),
            "bbpct":        round(float(r["BB%"]) if pd.notna(r["BB%"]) else 0, 1),
            "kbb":          round(float(r["KBB"])  if pd.notna(r["KBB"])  else 0, 2),
            "whip":         round(float(r["WHIP"]) if pd.notna(r["WHIP"]) else 0, 2),
            "ip":           round(float(r["IP"])   if pd.notna(r["IP"])   else 0, 1),
            "g":            int(r["G"])   if pd.notna(r["G"])   else 0,
            "bf":           int(r["BF"])  if pd.notna(r["BF"])  else 0,
            "k":            int(r["K"])   if pd.notna(r["K"])   else 0,
            "bb":           int(r["BB"])  if pd.notna(r["BB"])  else 0,
            "games_recent": int(r.get("games_recent", 0)),
            "ip_recent":    round(float(r.get("ip_recent", 0)), 1),
            "days_rested":  rest,
            "consec":       consec,
            "tier":         tier,
            "tier_color":   tier_color,
            "status":       status,
            "status_color": sc,
            "dots":         "".join(dots),
        })

    data_json = json.dumps(records)

    # dot labels for header (last 14 days)
    dot_labels = []
    for i in range(lookback - 1, -1, -1):
      d = today - timedelta(days=i + 1)                        # ← shift back one day
      dot_labels.append(f"{d.month}/{d.day}")
    dot_labels_json = json.dumps(dot_labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLB Reliever Dashboard {SEASON}</title>
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

  /* ── HEADER ── */
  .header {{ padding: 24px 28px 16px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 22px; font-weight: 600; letter-spacing: -0.3px; }}
  .header p  {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

  /* ── SUMMARY CARDS ── */
  .summary {{ display: flex; gap: 12px; padding: 16px 28px; flex-wrap: wrap; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
           padding: 12px 18px; min-width: 110px; }}
  .card-label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .card-val   {{ font-size: 22px; font-weight: 600; }}

  /* ── CONTROLS ── */
  .controls {{ display: flex; gap: 10px; padding: 12px 28px; flex-wrap: wrap; align-items: center; border-bottom: 1px solid var(--border); }}
  .controls input, .controls select {{
    background: var(--card); border: 1px solid var(--border); border-radius: 7px;
    color: var(--text); font-size: 12px; padding: 7px 11px; outline: none;
  }}
  .controls input {{ width: 200px; }}
  .controls input:focus, .controls select:focus {{ border-color: var(--accent); }}
  .controls label {{ font-size: 11px; color: var(--muted); }}
  .btn-group {{ display: flex; gap: 6px; margin-left: auto; }}
  .pill-btn {{
    font-size: 11px; padding: 5px 12px; border-radius: 20px; cursor: pointer;
    border: 1px solid var(--border); background: var(--card); color: var(--muted);
    transition: all .15s;
  }}
  .pill-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  /* ── TABLE ── */
  .table-wrap {{ overflow-x: auto; padding: 0 28px 28px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 14px; min-width: 900px; }}
  thead th {{
    font-size: 11px; font-weight: 500; color: var(--muted); text-align: left;
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; white-space: nowrap;
    position: sticky; top: 0; background: var(--bg); z-index: 2;
  }}
  thead th:hover {{ color: var(--text); }}
  thead th.sorted {{ color: var(--accent); }}
  thead th .arr {{ margin-left: 3px; opacity: .5; font-size: 10px; }}
  thead th.sorted .arr {{ opacity: 1; }}
  tbody tr {{ border-bottom: 1px solid var(--border); transition: background .1s; }}
  tbody tr:hover {{ background: var(--card); }}
  tbody td {{ padding: 9px 10px; vertical-align: middle; white-space: nowrap; }}

  /* ── CELLS ── */
  .name-cell {{ font-weight: 500; color: var(--text); }}
  .team-badge {{
    display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 7px;
    border-radius: 4px; background: var(--card); color: var(--muted);
    border: 1px solid var(--border); letter-spacing: .4px;
  }}
  .kpct-wrap {{ display: flex; align-items: center; gap: 7px; }}
  .kpct-bar-bg {{ width: 70px; height: 5px; border-radius: 3px; background: var(--border); flex-shrink: 0; }}
  .kpct-bar-fill {{ height: 5px; border-radius: 3px; }}
  .kpct-val {{ font-weight: 600; min-width: 38px; }}
  .tier-pill {{
    font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 4px;
    letter-spacing: .5px;
  }}
  .status-pill {{
    font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
    letter-spacing: .4px;
  }}
  .dots-wrap {{ display: flex; gap: 3px; align-items: center; }}
  .dot {{
    width: 9px; height: 9px; border-radius: 50%;
    flex-shrink: 0; cursor: default;
    position: relative;
  }}
  .dot:hover::after {{
    content: attr(data-tip);
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
    background: #1e2535; border: 1px solid var(--border); border-radius: 5px;
    padding: 3px 7px; font-size: 10px; white-space: nowrap; color: var(--text);
    pointer-events: none; z-index: 10;
  }}

  /* ── LEGEND ── */
  .legend {{ display: flex; gap: 18px; padding: 0 28px 0; flex-wrap: wrap; margin-bottom: 4px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--muted); }}
  .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; }}

  /* ── ROW COUNT ── */
  .row-count {{ font-size: 11px; color: var(--muted); padding: 6px 28px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>⚾ MLB Reliever Dashboard — {SEASON}</h1>
  <p>Statcast-derived K%, BB%, WHIP · Min {MIN_BF} batters faced · Recent window: last {lookback} days · Updated {today}</p>
</div>

<div class="summary" id="summaryCards"></div>

<div class="controls">
  <input type="text" id="search" placeholder="Search name or team..." oninput="render()" />
  <div>
    <label>Team&nbsp;</label>
    <select id="teamSel" onchange="render()"><option value="">All</option></select>
  </div>
  <div>
    <label>Min IP&nbsp;</label>
    <select id="minIP" onchange="render()">
      <option value="0">Any</option>
      <option value="5">5+</option>
      <option value="10">10+</option>
      <option value="20">20+</option>
    </select>
  </div>
  <div>
    <label>Recent&nbsp;</label>
    <select id="recentSel" onchange="render()">
      <option value="0">Any</option>
      <option value="1">Pitched today</option>
      <option value="2">Last 2 days</option>
      <option value="3">Last 3 days</option>
    </select>
  </div>
  <div class="btn-group" id="tierBtns">
    <button class="pill-btn active" data-tier="" onclick="setTier(this)">All tiers</button>
    <button class="pill-btn" data-tier="ELITE" onclick="setTier(this)">Elite</button>
    <button class="pill-btn" data-tier="GOOD"  onclick="setTier(this)">Good</button>
    <button class="pill-btn" data-tier="AVG"   onclick="setTier(this)">Avg</button>
    <button class="pill-btn" data-tier="BELOW" onclick="setTier(this)">Below</button>
  </div>
</div>

<div class="legend" style="margin-top:12px;">
  <span class="legend-item"><span class="legend-dot" style="background:#0F6E56"></span>Elite K% (35%+)</span>
  <span class="legend-item"><span class="legend-dot" style="background:#3B6D11"></span>Good (28–35%)</span>
  <span class="legend-item"><span class="legend-dot" style="background:#BA7517"></span>Avg (20–28%)</span>
  <span class="legend-item"><span class="legend-dot" style="background:#A32D2D"></span>Below (&lt;20%)</span>
  <span style="margin-left:12px" class="legend-item"><span class="legend-dot" style="background:#378ADD"></span>Pitched</span>
  <span class="legend-item"><span class="legend-dot" style="background:#2a3347;border:1px solid #4a5568"></span>Rest</span>
  <span class="legend-item"><span class="legend-dot" style="background:#A32D2D"></span>3rd consec day</span>
</div>

<div class="row-count" id="rowCount"></div>

<div class="table-wrap">
<table id="mainTable">
<thead>
<tr>
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
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<script>
const DATA       = {data_json};
const DOT_LABELS = {dot_labels_json};
const LOOKBACK   = {lookback};

let sortCol = "kpct", sortDir = -1, tierFilter = "";

function setTier(btn) {{
  document.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  tierFilter = btn.dataset.tier;
  render();
}}

// populate team dropdown
const teams = [...new Set(DATA.map(d => d.team))].sort();
const teamSel = document.getElementById("teamSel");
teams.forEach(t => {{ const o = document.createElement("option"); o.value = t; o.textContent = t; teamSel.appendChild(o); }});

function filtered() {{
  const q      = document.getElementById("search").value.toLowerCase();
  const team   = teamSel.value;
  const minIP  = parseFloat(document.getElementById("minIP").value);
  const recent = parseInt(document.getElementById("recentSel").value);

  return DATA.filter(d => {{
    if (q && !d.name.toLowerCase().includes(q) && !d.team.toLowerCase().includes(q)) return false;
    if (team && d.team !== team) return false;
    if (d.ip < minIP) return false;
    if (tierFilter && d.tier !== tierFilter) return false;
    if (recent > 0 && d.days_rested >= recent) return false;
    return true;
  }});
}}

function sorted(arr) {{
  return [...arr].sort((a, b) => {{
    let av = a[sortCol], bv = b[sortCol];
    if (typeof av === "string") return sortDir * av.localeCompare(bv);
    return sortDir * (bv - av);
  }});
}}

function dotHtml(dots) {{
  // dots is a string like "0010110..."
  let consec = 0;
  // find trailing consecutive 1s
  for (let i = dots.length - 1; i >= 0; i--) {{
    if (dots[i] === "1") consec++; else break;
  }}
  return dots.split("").map((v, i) => {{
    const label = DOT_LABELS[i] || "";
    const isLast = (dots.length - 1 - i) < consec;
    let bg;
    if (v === "0") bg = "#2a3347";
    else if (isLast && consec >= 3) bg = "#A32D2D";
    else bg = "#378ADD";
    return `<span class="dot" style="background:${{bg}}" data-tip="${{label}}"></span>`;
  }}).join("");
}}

function kpctBar(val, color) {{
  const w = Math.min(val / 45 * 100, 100).toFixed(1);
  return `<div class="kpct-wrap">
    <div class="kpct-bar-bg"><div class="kpct-bar-fill" style="width:${{w}}%;background:${{color}}"></div></div>
    <span class="kpct-val" style="color:${{color}}">${{val.toFixed(1)}}%</span>
  </div>`;
}}

function render() {{
  const rows = sorted(filtered());

  // summary cards
  const avgK = rows.length ? (rows.reduce((s,r) => s+r.kpct,0)/rows.length).toFixed(1) : "—";
  const elite = rows.filter(r => r.kpct >= 35).length;
  const fat   = rows.filter(r => r.consec >= 3).length;
  const fresh = rows.filter(r => r.days_rested >= 3).length;
  document.getElementById("summaryCards").innerHTML = `
    <div class="card"><div class="card-label">Relievers</div><div class="card-val">${{rows.length}}</div></div>
    <div class="card"><div class="card-label">Avg K%</div><div class="card-val">${{avgK}}%</div></div>
    <div class="card"><div class="card-label">Elite (35%+)</div><div class="card-val">${{elite}}</div></div>
    <div class="card"><div class="card-label">Fatigued (3+ consec)</div><div class="card-val" style="color:#A32D2D">${{fat}}</div></div>
    <div class="card"><div class="card-label">Fresh (3+ rest)</div><div class="card-val" style="color:#0F6E56">${{fresh}}</div></div>
  `;

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
      <td>${{r.kbb > 0 ? r.kbb.toFixed(2) : "—"}}</td>
      <td>${{r.whip > 0 ? r.whip.toFixed(2) : "—"}}</td>
      <td>${{r.ip.toFixed(1)}}</td>
      <td>${{r.g}}</td>
      <td>${{r.games_recent}}</td>
      <td style="color:${{r.days_rested <= 1 ? '#BA7517' : r.days_rested >= 3 ? '#0F6E56' : 'var(--text)'}}">${{r.days_rested === 99 ? "—" : r.days_rested + "d"}}</td>
      <td><span class="status-pill" style="background:${{r.status_color}}22;color:${{r.status_color}}">${{r.status}}</span></td>
      <td><div class="dots-wrap">${{dotHtml(r.dots)}}</div></td>
    </tr>
  `).join("");
}}

// sortable columns
document.querySelectorAll("thead th[data-col]").forEach(th => {{
  th.addEventListener("click", () => {{
    const col = th.dataset.col;
    if (sortCol === col) {{ sortDir *= -1; }}
    else {{ sortCol = col; sortDir = -1; }}
    document.querySelectorAll("thead th").forEach(h => {{
      h.classList.remove("sorted");
      const arr = h.querySelector(".arr");
      if (arr) arr.textContent = "↕";
    }});
    th.classList.add("sorted");
    const arr = th.querySelector(".arr");
    if (arr) arr.textContent = sortDir === -1 ? "↓" : "↑";
    render();
  }});
}});

render();
</script>
</body>
</html>"""
    return html


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    today = date.today()

    raw        = get_season_statcast(SEASON)
    rel        = classify_relievers(raw)
    print(f"Classifying relievers... {len(rel):,} reliever pitches")
    stats      = compute_stats(rel)
    print(f"Relievers with {MIN_BF}+ BF: {len(stats)}")
    recent     = get_recent(rel, LOOKBACK_DAYS)
    final      = stats.merge(recent, on="pitcher", how="left")
    final["games_recent"] = final["games_recent"].fillna(0).astype(int)
    final["days_rested"]  = final["days_rested"].fillna(99).astype(int)
    final["consec"]       = final["consec"].fillna(0).astype(int)
    final["ip_recent"]    = final["ip_recent"].fillna(0.0)
    final["app_dates"]    = final["app_dates"].apply(lambda x: x if isinstance(x, list) else [])

    html = build_html(final, LOOKBACK_DAYS, today)
    out = "index.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n✓ Dashboard saved → {out}")