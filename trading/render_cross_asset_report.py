"""Render CROSS_ASSET_REPORT.html from the experiment outputs.

Tables are read from the CSVs in ``outputs/`` so the page cannot drift
from the numbers; the five charts are embedded as data URIs.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from assets import UNIVERSE

PROJECT_DIR = Path(__file__).resolve().parent
OUT = PROJECT_DIR / "outputs"


def img(path):
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def name(key):
    return UNIVERSE[key]["name"]


def fmt(x, d=2):
    return f"{x:.{d}f}"


def atlas_table():
    t = pd.read_csv(OUT / "regime_atlas" / "regime_atlas.csv")
    rows = []
    for _, r in t.iterrows():
        early = r.t2_early_agreement
        late = r.t2_late_agreement
        def cell(v):
            cls = "pass" if v >= 0.9 else "fail"
            return f'<td class="n {cls}">{v*100:.0f}%</td>'
        rows.append(
            f"<tr><td>{name(r.key)}</td><td class='n'>{r.t2_occupancy_high*100:.0f}%</td>"
            f"<td class='n'>{r.t2_duration_low:.0f}</td><td class='n'>{r.t2_duration_high:.0f}</td>"
            f"<td class='n'>{r.t2_stdev_low:.2f}</td><td class='n'>{r.t2_stdev_high:.2f}</td>"
            f"<td class='n'>{r.t2_nu_low:.1f}</td><td class='n'>{r.t2_nu_high:.1f}</td>"
            f"<td class='n'>{int(r.t2_starts_within_1)}</td>{cell(early)}{cell(late)}</tr>"
        )
    return (
        "<div class='wrap'><table><thead><tr><th>Asset</th><th class='n'>Volatile share</th>"
        "<th class='n'>Calm days</th><th class='n'>Volatile days</th><th class='n'>Sd calm</th><th class='n'>Sd volatile</th>"
        "<th class='n'>nu calm</th><th class='n'>nu volatile</th><th class='n'>Starts of 12</th>"
        "<th class='n'>Early half</th><th class='n'>Late half</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def qlike_table():
    t = pd.read_csv(OUT / "vol_forecast_contest" / "qlike_r2_by_asset.csv").set_index("key")
    rows = []
    for key, r in t.iterrows():
        best = min(r.hmm, r.ewma, r.garch, r.har)
        def cell(v):
            cls = " best" if v == best else ""
            return f"<td class='n{cls}'>{v:.3f}</td>"
        rows.append(f"<tr><td>{name(key)}</td>{cell(r.hmm)}{cell(r.ewma)}{cell(r.garch)}{cell(r.har)}<td class='n'>{int(r.hmm_rank)}</td></tr>")
    return (
        "<div class='wrap'><table><thead><tr><th>Asset</th><th class='n'>HMM</th><th class='n'>EWMA</th>"
        "<th class='n'>GARCH-t</th><th class='n'>HAR-GK</th><th class='n'>HMM rank</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def mz_table():
    t = pd.read_csv(OUT / "vol_forecast_contest" / "mincer_zarnowitz.csv")
    rows = []
    for m, label in (("hmm", "HMM"), ("ewma", "EWMA"), ("garch", "GARCH-t")):
        s = t[t.model == m]
        sp = s[s.key == "sp500"].iloc[0]
        rows.append(f"<tr><td>{label}</td><td class='n'>{s.mz_slope.min():.2f} to {s.mz_slope.max():.2f}</td>"
                    f"<td class='n'>{s.mz_r2.min():.2f} to {s.mz_r2.max():.2f}</td><td class='n'>{sp.mz_r2:.2f}</td></tr>")
    return ("<table><thead><tr><th>Model</th><th class='n'>Slope range</th><th class='n'>R² range</th><th class='n'>S&P 500 R²</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def pooled_vt():
    p = pd.read_csv(OUT / "cross_asset_vol_target" / "pooled.csv")
    rows = []
    for _, r in p[p.cost == "base"].iterrows():
        u = {"futures": "13 futures", "etfs": "5 ETFs", "all": "All 18"}[r.universe]
        rows.append(f"<tr><td>{u}</td><td class='n'>{r.sharpe_hmm:.2f}</td><td class='n'>{r.sharpe_ewma:.2f}</td><td class='n'>{r.sharpe_garch:.2f}</td>"
                    f"<td class='n'>{r.sharpe_constant:.2f}</td><td class='n'>{r.diff_vs_ewma:+.2f}</td><td class='n'>{r.p_vs_ewma:.2f}</td><td class='n'>{r.ci_vs_ewma}</td></tr>")
    return ("<div class='wrap'><table><thead><tr><th>Universe</th><th class='n'>HMM</th><th class='n'>EWMA</th><th class='n'>GARCH</th><th class='n'>Constant</th>"
            "<th class='n'>HMM − EWMA</th><th class='n'>p</th><th class='n'>95% interval</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def cond_table():
    t = pd.read_csv(OUT / "regime_drift" / "conditional_means.csv")
    rows = []
    for _, r in t.iterrows():
        rows.append(f"<tr><td>{name(r.key)}</td><td class='n'>{r.mean_calm*252:+.1f}</td><td class='n'>{r.mean_volatile*252:+.1f}</td>"
                    f"<td class='n'>{r.share_calm*100:.0f}%</td><td class='n'>{r.t_stat:+.2f}</td><td class='n'>{r.p_value:.2f}</td></tr>")
    return ("<div class='wrap'><table><thead><tr><th>Asset</th><th class='n'>Calm, annualised</th><th class='n'>Volatile, annualised</th>"
            "<th class='n'>Share calm</th><th class='n'>t</th><th class='n'>p</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def pooled_generic(path, cols, labels):
    p = pd.read_csv(path)
    rows = []
    for _, r in p[p.cost == "base"].iterrows():
        u = {"futures": "13 futures", "etfs": "5 ETFs", "all": "All 18"}[r.universe]
        cells = "".join(f"<td class='n'>{r[c]:{f}}</td>" for c, f in cols)
        rows.append(f"<tr><td>{u}</td>{cells}</tr>")
    head = "".join(f"<th class='n'>{l}</th>" for l in labels)
    return f"<div class='wrap'><table><thead><tr><th>Universe</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def e6_counts():
    c = pd.read_csv(OUT / "cross_asset_regimes" / "counts.csv")
    lab = {"own_plus_equity_regime_vs_own": "S&P 500 regime probability", "own_plus_vix_vs_own": "Lagged log VIX", "own_plus_both_vs_own": "Both"}
    rows = [f"<tr><td>{lab[r.comparison]}</td><td class='n'>{int(r.improves)} of {int(r.assets)}</td><td class='n'>{int(r.improves_p5)}</td><td class='n'>{int(r.worsens_p5)}</td><td class='n'>{int(r.holm_p5)}</td></tr>"
            for _, r in c.iterrows()]
    return ("<table><thead><tr><th>Added regressor</th><th class='n'>Assets improved</th><th class='n'>Improved at 5%</th><th class='n'>Worsened at 5%</th><th class='n'>After Holm</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


HEAD = """<title>Eighteen Markets, One Regime Model</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --paper: #EEF1F3; --paper-2: #E2E7EA; --ink: #1A2229; --ink-2: #4B5760; --ink-3: #75828B;
    --rule: #CCD4D9; --accent: #1F6F73; --accent-soft: #D6E8E8; --fail: #B23A3A; --fail-soft: #F1DCDC;
    --pass: #2E7D5B; --warn: #A8720F; --warn-soft: #F3E7CC; --code-bg: #E4E9EC;
    --serif: "IBM Plex Serif", Georgia, "Times New Roman", serif;
    --cond: "IBM Plex Sans Condensed", "Arial Narrow", Arial, sans-serif;
    --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --paper: #151B20; --paper-2: #1D252C; --ink: #E6EAEC; --ink-2: #B2BCC4; --ink-3: #7E8A94;
    --rule: #303B44; --accent: #6FBFC2; --accent-soft: #1B3638; --fail: #E07A7A; --fail-soft: #3E2224;
    --pass: #6CC59A; --warn: #E0B65B; --warn-soft: #3A3014; --code-bg: #232C34; } }
  :root[data-theme="dark"] {
    --paper: #151B20; --paper-2: #1D252C; --ink: #E6EAEC; --ink-2: #B2BCC4; --ink-3: #7E8A94;
    --rule: #303B44; --accent: #6FBFC2; --accent-soft: #1B3638; --fail: #E07A7A; --fail-soft: #3E2224;
    --pass: #6CC59A; --warn: #E0B65B; --warn-soft: #3A3014; --code-bg: #232C34; }

  body { background: var(--paper); color: var(--ink); font-family: var(--serif); font-size: 17px; line-height: 1.55; }
  main { max-width: 820px; margin: 0 auto; padding: 40px 20px 80px; }
  h1, h2, h3 { font-family: var(--cond); font-weight: 600; letter-spacing: 0.01em; text-wrap: balance; margin: 0; }
  h1 { font-size: 2.5rem; line-height: 1.05; }
  h2 { font-size: 1.4rem; margin-top: 2.8rem; padding-top: 1rem; border-top: 2px solid var(--ink); }
  h2 small { font-family: var(--mono); font-size: 0.8rem; color: var(--ink-3); margin-left: 0.8rem; letter-spacing: 0.05em; }
  h3 { font-size: 1.05rem; margin-top: 1.5rem; }
  p { margin: 0.7rem 0; max-width: 68ch; }
  ul { max-width: 68ch; padding-left: 1.2rem; } li { margin: 0.35rem 0; }
  .eyebrow { font-family: var(--cond); font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .lede { font-size: 1.15rem; color: var(--ink-2); max-width: 62ch; margin-top: 0.9rem; }
  .meta { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-3); margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; }
  code, .num { font-family: var(--mono); font-size: 0.9em; font-variant-numeric: tabular-nums; }
  code { background: var(--code-bg); padding: 0.05em 0.35em; border-radius: 3px; }
  pre { background: var(--code-bg); padding: 0.8rem 1rem; border-radius: 4px; overflow-x: auto; font-family: var(--mono); font-size: 0.85rem; }

  .verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0; margin-top: 1.6rem; border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
  .verdict > div { padding: 0.9rem 1rem; background: var(--paper-2); border-right: 1px solid var(--rule); }
  .verdict > div:last-child { border-right: 0; }
  .verdict .k { font-family: var(--cond); font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
  .verdict .v { font-family: var(--mono); font-size: 1.5rem; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
  .verdict .s { font-size: 0.85rem; color: var(--ink-2); margin-top: 0.2rem; }
  .fail { color: var(--fail); } .pass { color: var(--pass); }

  .qa { display: grid; grid-template-columns: 6rem 1fr; gap: 0.3rem 1.2rem; margin-top: 1rem; }
  .qa .tag { font-family: var(--cond); font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); padding-top: 0.3rem; }
  .qa .tag.l { color: var(--accent); }
  .qa p { margin: 0.2rem 0 0.6rem; }
  .qa .lesson { font-weight: 600; }

  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 1rem 0; }
  th, td { text-align: left; padding: 0.4rem 0.55rem; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }
  th { font-family: var(--cond); font-weight: 600; font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
  td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
  td.best { font-weight: 600; color: var(--accent); }
  .wrap { overflow-x: auto; }

  figure { margin: 1.2rem 0 0; }
  figure img { width: 100%; height: auto; display: block; border: 1px solid var(--rule); border-radius: 3px; background: #fff; }
  figcaption { font-size: 0.85rem; color: var(--ink-2); margin-top: 0.4rem; max-width: 68ch; }

  .callout { border-left: 3px solid var(--accent); background: var(--accent-soft); padding: 0.8rem 1rem; margin: 1.2rem 0; max-width: 68ch; }
  .callout.warn { border-color: var(--warn); background: var(--warn-soft); }
  .callout p { margin: 0.3rem 0; }
  .learn { counter-reset: n; list-style: none; padding: 0; max-width: 68ch; }
  .learn li { counter-increment: n; display: grid; grid-template-columns: 2.2rem 1fr; gap: 0 0.6rem; padding: 0.8rem 0; border-top: 1px solid var(--rule); }
  .learn li::before { content: counter(n); font-family: var(--cond); font-size: 1.3rem; color: var(--accent); font-weight: 600; line-height: 1.2; }
  .learn b { display: block; font-family: var(--cond); font-size: 1rem; }
  @media (max-width: 560px) { .qa, .learn li { grid-template-columns: 1fr; } h1 { font-size: 1.9rem; } }
</style>
"""


def main():
    charts = {k: img(OUT / d / f) for k, d, f in (
        ("e2", "vol_forecast_contest", "01_qlike_by_asset.png"),
        ("e3", "cross_asset_vol_target", "01_vol_target.png"),
        ("e4", "regime_drift", "01_regime_drift.png"),
        ("e5", "tsmom_regime", "01_tsmom.png"),
        ("e6", "cross_asset_regimes", "01_cross_asset.png"),
    )}
    e4 = pooled_generic(OUT / "regime_drift" / "pooled.csv",
                        [("sharpe_constant", ".2f"), ("sharpe_regime_binary", ".2f"), ("sharpe_regime_scaled", ".2f"), ("sharpe_vol_managed", ".2f"),
                         ("diff_regime_binary_vs_constant", "+.2f"), ("p_regime_binary_vs_constant", ".2f"), ("diff_vol_managed_vs_constant", "+.2f"), ("p_vol_managed_vs_constant", ".2f")],
                        ["Constant", "Regime binary", "Regime scaled", "Vol-managed", "Binary − constant", "p", "Vol-managed − constant", "p"])
    e5 = pooled_generic(OUT / "tsmom_regime" / "pooled.csv",
                        [("sharpe_tsmom_ewma", ".2f"), ("sharpe_tsmom_hmm", ".2f"), ("sharpe_tsmom_garch", ".2f"), ("sharpe_tsmom_ewma_cut", ".2f"),
                         ("diff_tsmom_hmm", "+.2f"), ("p_tsmom_hmm", ".2f"), ("diff_tsmom_ewma_cut", "+.2f"), ("p_tsmom_ewma_cut", ".2f")],
                        ["EWMA sized", "HMM sized", "GARCH sized", "Cut in volatile", "HMM − EWMA", "p", "Cut − EWMA", "p"])

    body = f"""
<main>
<div class="eyebrow">HMM Regime Trading · cross-asset programme</div>
<h1>Eighteen Markets, One Regime Model</h1>
<p class="lede">The two-state Student-t regime model selected on WTI was fitted, walk-forward, to eighteen series across energy, metals, grains, equities, rates and FX, and asked six questions about forecasting and trading. It describes regimes well. It trades nothing.</p>
<div class="meta"><span>Run 2 September 2026</span><span>Scored 2015 to 2024, yearly refit</span><span>Filtered probabilities only</span><span>Holdouts sealed</span><span>Plan: RESEARCH_PROGRAMME.md</span></div>

<div class="verdict">
  <div><div class="k">Forecast contest</div><div class="v fail">4th of 4</div><div class="s">HMM loses to GARCH on pooled QLIKE, p = 0.016</div></div>
  <div><div class="k">Vol target vs EWMA</div><div class="v">−0.02</div><div class="s">Sharpe, 13 futures pooled, p = 0.61</div></div>
  <div><div class="k">Calm-state drift</div><div class="v">0 of 18</div><div class="s">assets with a real-time premium</div></div>
  <div><div class="k">Per-asset tests &lt; 5%</div><div class="v">5 of 162</div><div class="s">8.1 expected by chance; none survive Holm</div></div>
</div>

<div class="callout"><p><b>Status.</b> Everything on this page is exploratory, run on the development sample only. No threshold was searched, no model was reselected, and no holdout was opened. A positive result would have become a candidate for a dated gate entry. None did.</p></div>

<h2>Why widen the scope <small>context</small></h2>
<p>The registered WTI test found the regime signal no better than a 21-day realised volatility at sizing a position, and the whole margin sat in two days of April 2020. One asset and one rule cannot separate a fact about the model from a fact about WTI. The programme below ran the same question on eighteen series and added the questions the first test could not ask: whether the model forecasts variance at all, whether the calm state carries a return premium, whether the regime helps a directional rule, and whether one market's regime informs another's.</p>
<p>The universe is thirteen Yahoo continuous futures (WTI, Brent, Henry Hub, gold, silver, copper, corn, soybeans, wheat, S&amp;P 500, 10-year note, euro, yen) and five ETFs as a roll-free, dividend-adjusted control (SPY, TLT, GLD, USO, UNG). Inputs are percent log returns, except WTI, which keeps the research decision of dollar changes because its price crossed zero.</p>

<h2>E1. Regime atlas <small>05_regime_atlas.py</small></h2>
<div class="qa">
  <div class="tag">Question</div><p>Does the WTI finding, two persistent volatility states with a heavy-tailed volatile state and near-zero means, hold elsewhere?</p>
  <div class="tag">Answer</div><p>Mostly. Student-t beats Gaussian on BIC in all eighteen series, every fit converges from at least eleven of twelve starts, and the volatile state is a broad regime (28% to 52% of days, durations from a month to a year), not an April 2020 bucket. Sample stability is the dividing line: eight series give the same state path whichever half of the sample they are fitted on; the metals and USO do not.</p>
</div>
{atlas_table()}
<figcaption>Units are percent per day, WTI in dollars a barrel. Green cells meet the research 90% agreement bar between a half-sample fit and the full fit; red cells miss it.</figcaption>
<div class="qa"><div class="tag l">Lesson</div><p class="lesson">Initialisation stability is a property of the model; sample stability is a property of the asset. The same estimator that is rock-solid on the S&amp;P 500 finds a different regime map for gold depending on which years it sees.</p></div>

<h2>E2. Volatility forecasting contest <small>06_vol_forecast_contest.py</small></h2>
<div class="qa">
  <div class="tag">Question</div><p>The HMM is a volatility model. Before asking whether it trades, how well does it forecast tomorrow's variance next to EWMA, a GARCH(1,1) with Student-t innovations, and a HAR on the Garman-Klass range?</p>
  <div class="tag">Answer</div><p>Worst of the four in most assets. On the pooled QLIKE against the squared return, GARCH beats it (p = 0.016) and EWMA edges it (p = 0.086). Against the range proxy it loses to every rival at p &lt; 0.001. Its only "win", against HAR on the squared-return proxy, is a bias effect: HAR forecasts the range variance, which is smaller than the close-to-close variance.</p>
</div>
<figure><img src="{charts['e2']}" alt="Mean QLIKE per asset and model"><figcaption>Mean QLIKE loss, 2015 to 2024, squared-return proxy. Lower is better. WTI in dollar units.</figcaption></figure>
{qlike_table()}
<h3>The diagnosis</h3>
<p>A Mincer-Zarnowitz regression of the squared return on each forecast separates calibration from information:</p>
{mz_table()}
<div class="qa"><div class="tag l">Lesson</div><p class="lesson">Slope near one, R² near zero. The regime forecast is unbiased and almost uninformative day to day. Two persistent states give a forecast that moves slowly between two levels, so it explains the level of volatility over months and none of the change from one day to the next. GARCH tracks both. This single table explains every trading null below.</p></div>

<h2>E3. Volatility targeting across assets <small>07_cross_asset_vol_target.py</small></h2>
<div class="qa">
  <div class="tag">Question</div><p>The registered WTI test on every asset: target one unit of daily volatility, cap three, costs of one and two hundredths per unit of turnover, HMM sizing against EWMA sizing.</p>
  <div class="tag">Answer</div><p>The WTI null is general. Six of eighteen differences are positive, none below 5%. On the roll-free ETFs the HMM sizer is significantly worse than EWMA. WTI's +0.12 is still the largest entry, still the two April 2020 days, still not significant.</p>
</div>
{pooled_vt()}
<figure><img src="{charts['e3']}" alt="Vol-target portfolio and per-asset Sharpe differences"><figcaption>Top: equal-risk portfolio of thirteen futures, cumulative net P&amp;L at base costs. Bottom: HMM minus EWMA Sharpe per asset with 95% block-bootstrap intervals. Every interval covers zero.</figcaption></figure>

<h2>E4. Regime drift <small>08_regime_drift.py</small></h2>
<div class="qa">
  <div class="tag">Question</div><p>Does the calm state carry the risk premium? The in-sample atlas says the S&amp;P 500 returns +0.12% a day in its calm state and −0.03% in its volatile state.</p>
  <div class="tag">Answer</div><p>Not in real time. Sorting days by the one-step predicted state, no asset shows a significant drift difference, and in equities the sign flips: predicted-volatile days returned more. The smoothed state means use the whole sample, including the return on the day being labelled. The one-step probability does not.</p>
</div>
{cond_table()}
<figure><img src="{charts['e4']}" alt="Regime drift portfolio and conditional means"><figcaption>Top: long-only rules on the futures portfolio, each asset scaled to unit volatility. Bottom: annualised mean return by predicted state.</figcaption></figure>
{e4}
<p>The one nominally significant line in the pooled table is a published result, not the model: the Moreira-Muir volatility-managed rule on the five ETFs beats constant long by 0.36 Sharpe units (p = 0.04, one of six universe-and-cost lines, not significant after Holm). The regime rules never beat it.</p>
<div class="qa"><div class="tag l">Lesson</div><p class="lesson">Smoothed state statistics are not tradable statistics. Any regime study that reports state-conditional returns from the smoothed path is reporting look-ahead.</p></div>

<h2>E5. Momentum with a regime overlay <small>09_tsmom_regime.py</small></h2>
<div class="qa">
  <div class="tag">Question</div><p>Time-series momentum (sign of the past 252 days, sized to unit volatility) sized by EWMA, by the HMM, or by EWMA with the position cut to zero when the volatile state is more likely than not.</p>
  <div class="tag">Answer</div><p>HMM sizing is worse than EWMA sizing on the futures portfolio (−0.06, p = 0.82). The cut removes a quarter of the portfolio's volatility and leaves its Sharpe where it was: it takes out good and bad days alike. Copper's +0.34 is the one per-asset hit in eighteen.</p>
</div>
{e5}
<figure><img src="{charts['e5']}" alt="Momentum variants"><figcaption>Top: futures portfolio, four momentum variants. Bottom: Sharpe per asset at base costs.</figcaption></figure>
<div class="qa"><div class="tag l">Lesson</div><p class="lesson">The momentum-crash story is about equity cross-sectional momentum after bear markets. A one-day-ahead volatility state on a single futures contract is a different object and does not carry it.</p></div>

<h2>E6. Cross-asset regime information <small>10_cross_asset_regimes.py</small></h2>
<div class="qa">
  <div class="tag">Question</div><p>Do regimes co-move, and does another market's regime improve an asset's own variance forecast beyond what a public volatility index already gives?</p>
  <div class="tag">Answer</div><p>They co-move in clusters (gold and silver 0.79, corn and soybeans 0.67, WTI and the S&amp;P 500 0.47) and the equity regime leads WTI's by about four days. Adding the equity regime to an asset's own variance regression helps crude and gas a little and hurts the euro and corn. Adding yesterday's VIX helps sixteen of eighteen and never hurts.</p>
</div>
{e6_counts()}
<figure><img src="{charts['e6']}" alt="Regime correlations and lead-lag"><figcaption>Left: correlation of one-step high-state probabilities. Right: correlation of each asset's regime with the S&amp;P 500 regime shifted by up to ten days.</figcaption></figure>
<div class="qa"><div class="tag l">Lesson</div><p class="lesson">Cross-asset regime information is real and the VIX already carries it. A shared regime factor would have to beat a free public index, which is the bar for the research Stage 5 multivariate model.</p></div>

<h2>What was learned</h2>
<ol class="learn">
  <li><b>The regime model is a description, not a forecaster.</b>It says which of two volatility levels a market is in, with lags of weeks, and says so reliably in most assets. It does not say what tomorrow's variance will be any better than EWMA, and worse than a four-parameter GARCH.</li>
  <li><b>Smoothed is not filtered.</b>The in-sample split of equity returns by state (calm about +30% a year, volatile about −8%) vanishes in real time. The difference between the two is exactly the information the trader does not have.</li>
  <li><b>The pipeline replicates what it should.</b>Volatility management improves the ETF portfolio, time-series momentum earns a modest positive Sharpe on futures, Student-t beats Gaussian everywhere. The nulls are about the signal, not the code.</li>
  <li><b>Count the tests.</b>162 per-asset strategy tests produced five p-values under 5% against 8.1 expected. Read without the count line, the tables would have "found" copper momentum and euro volatility targeting.</li>
  <li><b>Where a regime might still matter</b> is not sizing: options, where state duration maps to a horizon; and a within-state GARCH, a regime model that also tracks day-to-day variance. Each needs its own dated gate entry before any run.</li>
</ol>

<h2>Limitations</h2>
<ul>
  <li>Yahoo continuous futures are front-month splices with roll gaps (corn −27% and natural gas +38% in a day are rolls). The ETF control handles this for five series; every conclusion holds on the ETFs as well.</li>
  <li>Costs are one and two hundredths of a percent of notional per unit of turnover, a futures assumption applied to ETFs too.</li>
  <li>Range variance is missing on 701 silver days and 202 gold days with zero recorded range; HAR falls back to the squared return there.</li>
  <li>Pooled portfolios weight assets by their own volatility target, not a covariance estimate. They are a test device.</li>
  <li>Thresholds (one half, 252 days, 21-day span, 3-unit cap) were fixed in the programme document before any run, but by convention, not by registration against alternatives.</li>
</ul>

<h2>Reproduction</h2>
<pre>python3 03_download_assets.py     # refuses to overwrite frozen files
python3 04_fit_signals.py         # about 10 minutes on 24 cores
python3 05_regime_atlas.py
python3 06_vol_forecast_contest.py
python3 07_cross_asset_vol_target.py
python3 08_regime_drift.py
python3 09_tsmom_regime.py
python3 10_cross_asset_regimes.py
python3 render_cross_asset_report.py</pre>
<p class="meta">Folder: trading · full tables in CROSS_ASSET_RESULTS.md and outputs/</p>
</main>
"""
    (PROJECT_DIR / "CROSS_ASSET_REPORT.html").write_text(HEAD + body)
    print("written", (PROJECT_DIR / "CROSS_ASSET_REPORT.html").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
