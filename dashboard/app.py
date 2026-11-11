"""
Thiruvananthapuram green-loss dashboard (Step 7).

    .venv/Scripts/streamlit run dashboard/app.py

Data comes from dashboard/data (built by scripts/step7_prepare.py). The live spatial query can run
against the PostGIS database from Step 5b (schema tvm) or, without a database, on the local pixel table.
The database password is read from PGPASSWORD / pgpass.conf, or typed into the sidebar; it is kept only
in this browser session's memory and never written anywhere.
"""
import json
import os
import re
import time
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shapely
import streamlit as st
from branca.colormap import LinearColormap
from folium import plugins
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from streamlit_folium import st_folium

DATA = Path(__file__).resolve().parent / "data"
TAB = DATA / "tables"
PX_KM2 = 900 / 1e6
BASE_TILES = "Esri.WorldGrayCanvas"          # keyless light base map (CartoDB now needs an API key)
GREEN = (2, 3)
CENTER = (8.505, 76.94)
YEARS = [2000, 2010, 2017, 2025]
TO_UTM = Transformer.from_crs(4326, 32643, always_xy=True).transform
TO_WGS = Transformer.from_crs(32643, 4326, always_xy=True).transform

st.set_page_config(page_title="TVM Green Loss", page_icon="🌳", layout="wide")

# ---------------------------------------------------------------- look and feel
# Colours and fonts live in .streamlit/config.toml; this CSS only adds cards, the hero banner and legend chips.
INK, MUTED, LINE, CARD = "#1c2a22", "#5b6b61", "#d9e2d5", "#ffffff"
st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem; max-width: 1440px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
[data-testid="stMetric"] {{background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 18px 12px;
    box-shadow: 0 1px 2px rgba(20,58,42,.05), 0 6px 16px rgba(20,58,42,.05); height: 100%;}}
[data-testid="stMetricLabel"] p {{font-size: .86rem; color: {MUTED}; font-weight: 500;}}
[data-testid="stMetricValue"] {{font-weight: 700; color: #143a2a; font-size: clamp(1.5rem, 2.2vw, 2.1rem);}}
[data-testid="stMetric"] div, [data-testid="stMetric"] p {{white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important;}}
[data-testid="stPlotlyChart"] {{background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 6px 4px 0;}}
[data-testid="stImage"] img, iframe {{border-radius: 12px;}}
[data-testid="stMain"] [data-testid="stExpander"] details {{background: {CARD};}}
.hero {{background: linear-gradient(120deg, #143a2a 0%, #1f7a4d 55%, #5aa86f 100%); color: #fff;
    border-radius: 20px; padding: 30px 34px 28px; margin: 0 0 22px; box-shadow: 0 10px 30px rgba(20,58,42,.18);}}
.hero .kicker {{text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; opacity: .8; margin-bottom: 6px;}}
.hero h1 {{color: #fff; font-size: 2.3rem; margin: 0 0 10px; padding: 0; line-height: 1.15;}}
.hero p {{color: #e3f2e7; font-size: 1.12rem; margin: 0; line-height: 1.6; max-width: 62rem;}}
.hero b {{color: #fff; background: rgba(255,255,255,.16); padding: 1px 7px; border-radius: 6px; white-space: nowrap;}}
.page-head {{margin: 0 0 14px;}}
.page-head h1 {{margin: 0; padding: 0 0 4px; font-size: 2rem;}}
.page-head p {{margin: 0; color: {MUTED}; font-size: 1.02rem;}}
.howto {{background: {CARD}; border: 1px solid {LINE}; border-left: 5px solid #1f7a4d; border-radius: 12px;
    padding: 12px 16px; margin: 0 0 18px; color: #33443a; line-height: 1.55;}}
.chips {{display: flex; flex-wrap: wrap; gap: 6px 8px; margin: 8px 0 4px; font-size: .84rem;}}
.chip {{display: inline-flex; align-items: center; gap: 6px; background: {CARD}; border: 1px solid {LINE};
    border-radius: 999px; padding: 3px 11px 3px 5px; color: #33443a;}}
.chip i {{width: 14px; height: 14px; border-radius: 50%; display: inline-block; box-shadow: inset 0 0 0 1px rgba(0,0,0,.18);}}
</style>""", unsafe_allow_html=True)

pio.templates["tvm"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, sans-serif", size=13, color="#33443a"),
    title=dict(font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=16, color="#143a2a"), x=0.02,
               xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", barcornerradius=4,
    hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(color=INK, size=13, family="Inter, sans-serif")),
    xaxis=dict(showgrid=False, linecolor="#c3cec0", ticks="", title_font=dict(color=MUTED)),
    yaxis=dict(gridcolor="rgba(28,42,34,0.08)", zeroline=False, title_font=dict(color=MUTED))))
pio.templates.default = "plotly_white+tvm"

# ---------------------------------------------------------------- data


@st.cache_data
def load_geo(name):
    return gpd.read_file(DATA / f"{name}.geojson")


@st.cache_data
def load_tab(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_overlays():
    return json.loads((DATA / "overlays.json").read_text())


@st.cache_data
def load_px():
    return pd.read_parquet(DATA / "px30.parquet")


wards = load_geo("wards")
OV = load_overlays()
LC_LEGEND, CHANGE_LEGEND = OV["_legend"]["landcover"], OV["_legend"]["change"]

WARD_METRICS = {
    "green_change_pp_2000_2025": "Green change 2000-2025 (pp)",
    "green_change_pp_2017_2025": "Green change 2017-2025 (pp)",
    "green_pct_2025": "Green cover 2025 (%)",
    "built_pct_2025": "Built-up 2025 (%)",
    "green_to_built_km2": "Green -> built-up 2000-2025 (km2)",
    "priority_index": "Priority index (equal weights)",
    "green_per_capita_2025": "Green per person 2025 (m2)",
    "density_2025_per_km2": "Population density 2025 (/km2)",
    "modis_lst_2025": "Dry-season LST 2025 (C)",
    "building_coverage_pct": "Building footprint cover (%)",
    "road_density_km_per_km2": "Road density (km/km2)",
}
REVERSED = {"green_change_pp_2000_2025", "green_change_pp_2017_2025", "green_pct_2025", "green_per_capita_2025"}
INDICATORS = {
    "green_loss_pp": "Green loss 2000-2025 (pp)",
    "recent_loss_pp": "Green loss 2017-2025 (pp)",
    "hot_share_pct": "Area in Gi* hotspots (%)",
    "lowland_loss_pct": "Low-lying green converted (%)",
    "lst_2025": "Dry-season LST 2025 (C)",
    "green_per_capita": "Green per person 2025 (m2, less = worse)",
    "density": "Population density 2025",
}
PRESETS = {
    "Equal weights": dict.fromkeys(INDICATORS, 1),
    "Loss pressure": {"green_loss_pp": 3, "recent_loss_pp": 3, "hot_share_pct": 2, "lowland_loss_pct": 1,
                      "lst_2025": 1, "green_per_capita": 1, "density": 1},
    "Climate risk": {"green_loss_pp": 1, "recent_loss_pp": 1, "hot_share_pct": 1, "lowland_loss_pct": 3,
                     "lst_2025": 3, "green_per_capita": 1, "density": 1},
    "People": {"green_loss_pp": 1, "recent_loss_pp": 1, "hot_share_pct": 1, "lowland_loss_pct": 1,
               "lst_2025": 1, "green_per_capita": 3, "density": 3},
}
GI_COLORS = {"Hot 99%": "#b2182b", "Hot 95%": "#ef8a62", "Hot 90%": "#fddbc7", "Not significant": "#f0f0f0",
             "Cold 90%": "#d1e5f0", "Cold 95%": "#67a9cf", "Cold 99%": "#2166ac"}


def ward_label(no):
    return f"{no} · {wards.set_index('ward_no').ward_name[no]}"


def legend(items, cols=4):
    html = "".join(f"<span class='chip'><i style='background:{c}'></i>{n.replace('->', '→')}</span>"
                   for n, c in items.items())
    st.markdown(f"<div class='chips'>{html}</div>", unsafe_allow_html=True)


def md_inline(text):
    """**bold** and *italic* to HTML, for text placed inside our own HTML blocks."""
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text))


def head(icon, title, subtitle):
    st.markdown(f"<div class='page-head'><h1>{icon} {title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def base_map(zoom=12, height_center=CENTER, tiles=BASE_TILES):
    m = folium.Map(location=height_center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(tiles, name="Light base map", max_zoom=16).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite (Esri)", show=False).add_to(m)
    return m


def add_overlay(m, key, show=True, opacity=0.85):
    o = OV[key]
    folium.raster_layers.ImageOverlay(image=str(DATA / "img" / f"{key}.png"), bounds=o["bounds"],
                                      opacity=opacity, name=o["label"], show=show, interactive=False,
                                      zindex=2).add_to(m)


def add_outline(m):
    folium.GeoJson(load_geo("corporation"), name="Corporation boundary",
                   style_function=lambda f: {"color": "#222", "weight": 2, "fill": False},
                   control=False).add_to(m)


def ward_layer(m, metric=None, name="Wards", show=True, highlight=None):
    if metric:
        v = wards[metric]
        cm = LinearColormap(["#1a9850", "#fee08b", "#d73027"] if metric not in REVERSED
                            else ["#d73027", "#fee08b", "#1a9850"],
                            vmin=float(v.quantile(0.02)), vmax=float(v.quantile(0.98)),
                            caption=WARD_METRICS.get(metric, metric))
        style = lambda f: {"fillColor": cm(min(max(f["properties"][metric], cm.vmin), cm.vmax))  # noqa: E731
                           if f["properties"][metric] is not None else "#ccc",
                           "color": "#444", "weight": 0.6, "fillOpacity": 0.7}
        cm.add_to(m)
    else:
        style = lambda f: {"color": "#555", "weight": 0.8, "fill": True, "fillOpacity": 0.0}  # noqa: E731
    fields = ["ward_no", "ward_name", "green_pct_2000", "green_pct_2025", "green_change_pp_2000_2025",
              "priority_rank"] + ([metric] if metric and metric not in ("green_change_pp_2000_2025",) else [])
    aliases = ["Ward no", "Ward", "Green 2000 %", "Green 2025 %", "Green change pp", "Priority rank"] + \
              ([WARD_METRICS.get(metric, metric)] if metric and metric != "green_change_pp_2000_2025" else [])
    gj = folium.GeoJson(
        wards[fields + ["geometry"]].to_json(), name=name, show=show, style_function=style,
        highlight_function=lambda f: {"weight": 3, "color": "#000"},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True))
    gj.add_to(m)
    if highlight is not None:
        folium.GeoJson(wards[wards.ward_no == highlight][["geometry"]].to_json(), control=False,
                       style_function=lambda f: {"color": "#00e5ff", "weight": 4, "fill": False}).add_to(m)
    return gj


def clicked_ward(out):
    f = (out or {}).get("last_active_drawing") or {}
    props = f.get("properties") or {}
    return props.get("ward_no")


def plot(fig, height=None):
    fig.update_layout(margin=dict(l=12, r=16, t=56, b=12), height=height,
                      legend=dict(orientation="h", y=-0.2, title=None))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def intro(text):
    """One plain-English line telling the reader what the page shows and what to look for."""
    st.markdown(f"<div class='howto'>💡 <b>How to read this page.</b> {md_inline(text)}</div>",
                unsafe_allow_html=True)


PITCH_KM2 = 0.00714   # one football pitch (105 x 68 m), to make areas tangible


# ---------------------------------------------------------------- sidebar

st.sidebar.title("🌳 TVM Green Loss")
st.sidebar.caption("Urban expansion and green space loss, Thiruvananthapuram Corporation, 2000-2025")
PAGE_ICONS = {"Overview": "🏠", "Map explorer": "🗺️", "Ward drill-down": "🔎", "Hotspots & rules": "🔥",
              "Spatial query": "✏️", "Priority (decision support)": "🎯", "Big data engine": "⚙️"}
PAGES = list(PAGE_ICONS)
if "page" not in st.session_state and st.query_params.get("page") in PAGES:   # deep link: ?page=Map explorer
    st.session_state.page = st.query_params["page"]
page = st.sidebar.radio("Explore", PAGES, key="page", format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")
if "ward" not in st.session_state:
    st.session_state.ward = int(wards.sort_values("priority_rank").ward_no.iloc[0])
with st.sidebar.expander("How to read this dashboard"):
    st.markdown("- **Green** = trees + other vegetation (fields, gardens, scrub).\n"
                "- **Built-up** = roofs, roads and paved land.\n"
                "- **pp** = percentage points: 40% → 35% is −5 pp.\n"
                "- Maps come from satellite images; each square pixel is 30 m wide.\n"
                "- 2000-2017 maps are estimated back from 2025, so treat small changes with care.")
st.sidebar.divider()
st.sidebar.caption("Data: Landsat 5/7/8/9 and Sentinel-2 (Google Earth Engine), K-SMART 2024 wards, "
                   "OpenStreetMap, WorldPop, MERIT Hydro, MODIS. Land cover by Random Forest; "
                   "1 Landsat pixel = 30 m.")

# ---------------------------------------------------------------- pages


def page_overview():
    cs = load_tab("change_summary").set_index("period")
    trend = load_tab("trend")
    full = cs.loc["2000-2025"]
    t = trend.pivot(index="year", columns="class", values="pct")
    green = t[["Tree cover", "Other vegetation"]].sum(axis=1)
    heat = load_tab("heat_landsat").set_index("group")
    case = load_tab("case_studies").set_index("metric").value
    pitches = full.green_to_built_km2 / PITCH_KM2
    st.markdown(f"<div class='hero'><div class='kicker'>Thiruvananthapuram Corporation · 2000 → 2025</div>"
                f"<h1>🌳 Is the city losing its green?</h1>"
                f"<p><b>Yes.</b> In 25 years, <b>{full.green_to_built_km2:.1f} km²</b> of trees and fields were "
                f"built over — about <b>{pitches:,.0f} football pitches</b>, or one every "
                f"<b>{25 * 365 / pitches:.0f} days</b>.</p></div>", unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("🌿 Green cover today", f"{green[2025]:.0f}%", f"{green[2025] - green[2000]:+.1f} pp since 2000",
                help="Share of the corporation covered by trees or other vegetation in 2025.")
    c[1].metric("🏠 Built-up today", f"{t.loc[2025, 'Built-up']:.0f}%",
                f"{t.loc[2025, 'Built-up'] - t.loc[2000, 'Built-up']:+.1f} pp since 2000", delta_color="inverse",
                help="Share covered by roofs, roads and paved land in 2025.")
    c[2].metric("🌡️ Extra heat on lost green", f"+{heat.loc['Green -> built-up', 'extra_warming_vs_stable_green_c']:.1f} °C",
                "vs land that stayed green", delta_color="off", delta_arrow="off",
                help="Satellite surface temperature rose this much more on converted land than on stable green land.")
    c[3].metric("👤 Green per person",
                f"{case['Corporation green m2 per person, 2025 (WorldPop R2025A)']:.0f} m²",
                f"{case['Corporation green m2 per person, 2025 (WorldPop R2025A)'] - case['Corporation green m2 per person, 2017 (WorldPop R2025A)']:+.0f} m² since 2017",
                help="Green area divided by population (WorldPop). Compared only from 2017, within one data release.")

    st.subheader("🛰️ See it: the city in 2000 and 2025")
    st.caption("Red = buildings and roads, dark green = trees, light green = other vegetation. "
               "The right-hand map shows only the land that turned from green to built-up.")
    i1, i2, i3 = st.columns(3)
    i1.image(str(DATA / "img" / "lc_2000.png"), caption="2000", width="stretch")
    i2.image(str(DATA / "img" / "lc_2025.png"), caption="2025", width="stretch")
    i3.image(str(DATA / "img" / "green_to_built.png"), caption="Green lost to buildings, 2000-2025", width="stretch")
    legend({k: v for k, v in LC_LEGEND.items() if k != "Water"})
    st.caption("Want to zoom in? Open **Map explorer** and choose *Compare two years*.")

    left, right = st.columns(2)
    with left:
        g = pd.DataFrame({"year": YEARS * 2, "pct": list(green[YEARS]) + list(t.loc[YEARS, "Built-up"]),
                          "what": ["Green"] * 4 + ["Built-up"] * 4})
        fig = px.line(g, x="year", y="pct", color="what", markers=True,
                      color_discrete_map={"Green": "#1a7837", "Built-up": "#d7301f"},
                      title="Green shrinks as buildings spread (% of the city)")
        fig.update_traces(line_width=3, marker_size=9, hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
        for name, s in (("Green", green), ("Built-up", t["Built-up"])):
            fig.add_annotation(x=2025, y=s[2025], text=f"<b>{name} {s[2025]:.0f}%</b>", showarrow=False,
                               xanchor="left", xshift=8)
            fig.add_annotation(x=2000, y=s[2000], text=f"{s[2000]:.0f}%", showarrow=False, xanchor="right", xshift=-8)
        fig.update_xaxes(tickvals=YEARS, range=[1996, 2033], title=None)
        fig.update_yaxes(title=None, ticksuffix="%", rangemode="tozero")
        fig.update_layout(showlegend=False)
        plot(fig, 380)
    with right:
        w = wards.nsmallest(10, "green_change_pp_2000_2025").sort_values("green_change_pp_2000_2025")
        w = w.assign(loss=-w.green_change_pp_2000_2025)   # plot loss as a positive length so labels sit right of the bar
        fig = px.bar(w, x="loss", y="ward_name", orientation="h",
                     text=w.loss.map(lambda v: f"−{v:.0f} pp"),
                     custom_data=["green_pct_2000", "green_pct_2025"],
                     title="The 10 wards that lost the most green")
        fig.update_traces(marker_color="#d7301f", textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{y}</b><br>green %{customdata[0]:.0f}% → %{customdata[1]:.0f}%"
                                        "<extra></extra>")
        fig.update_yaxes(title=None, autorange="reversed")
        fig.update_xaxes(title="green cover lost, 2000-2025 (percentage points)", range=[0, w.loss.max() * 1.15])
        plot(fig, 380)

    st.subheader("🧭 Why does green disappear?")
    st.caption("Three patterns explain most of the loss: roads, heat and infill.")
    rd = load_tab("road_distance")
    hs = load_tab("hotspot_summary").set_index("metric").value
    rules = load_tab("rules_conversion")
    rs = load_tab("rules_summary").iloc[0]
    top = rules.iloc[0]
    a, b, c3 = st.columns(3)
    with a:
        fig = px.bar(rd, x="band", y="conversion_rate_pct", text=rd.conversion_rate_pct.map(lambda v: f"{v:.0f}%"),
                     title="Near main roads: 5× more loss")
        fig.update_traces(marker_color="#d7301f", textposition="outside", cliponaxis=False,
                          hovertemplate="%{x} from a main road: %{y:.1f}% of green converted<extra></extra>")
        fig.update_xaxes(title="distance from a main road")
        fig.update_yaxes(title="% of 2000 green built over", ticksuffix="%")
        plot(fig, 340)
    with b:
        hd = heat.loc[["Stable green", "Green -> built-up"]].reset_index()
        hd["group"] = ["Stayed green", "Green → built-up"]
        fig = px.bar(hd, x="group", y="delta_c", color="group", text=hd.delta_c.map(lambda v: f"+{v:.1f} °C"),
                     color_discrete_map={"Stayed green": "#1a7837", "Green → built-up": "#d7301f"},
                     title="Built-over land: 2× the warming")
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}: surface warmed %{y:.1f} °C, 2000-2025<extra></extra>")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="surface temperature rise, 2000-2025", ticksuffix=" °C")
        fig.update_layout(showlegend=False)
        plot(fig, 340)
    with c3:
        with st.container(border=True):
            st.markdown("##### 🧱 Building fills in the gaps")
            st.markdown(f"Green land **already surrounded by buildings** in 2000 was built over "
                        f"**{100 * top.confidence:.0f}%** of the time — **{top.lift:.1f}×** the city average of "
                        f"{rs.base_conversion_rate_pct:.0f}%.")
        with st.container(border=True):
            st.markdown("##### 📍 Loss is concentrated")
            st.markdown(f"**{hs['Green->built km2 in hot hexagons (p<=0.05)']:.1f} km²** of the loss sits in a few "
                        "hotspots, mainly the belt from Pettah and Chakkai to Sreevaraham and Kalippankulam. "
                        "See **Hotspots & rules**.")

    with st.expander("Details: where every km² of land went, and numbers by period"):
        tr = load_tab("transitions")
        period = st.selectbox("Transition period", tr.period.unique().tolist(),
                              index=tr.period.unique().tolist().index("2000-2025"))
        d = tr[(tr.period == period) & (tr.km2 > 0.05)]
        names = list(LC_LEGEND)
        pa, pb = period.split("-")
        fig = go.Figure(go.Sankey(
            node=dict(label=[f"{n} {pa}" for n in names] + [f"{n} {pb}" for n in names],
                      color=list(LC_LEGEND.values()) * 2, pad=12),
            link=dict(source=d.from_class - 1, target=d.to_class - 1 + len(names), value=d.km2,
                      color=["rgba(215,48,31,0.45)" if (f in (2, 3) and t_ == 1) else "rgba(160,160,160,0.3)"
                             for f, t_ in zip(d.from_class, d.to_class)],
                      hovertemplate="%{source.label} → %{target.label}: %{value:.2f} km²<extra></extra>")))
        fig.update_layout(title=f"Where land went, {period} (red = green → built-up)")
        plot(fig, 360)
        show = cs.reset_index()[["period", "years", "green_start_km2", "green_end_km2", "green_lost_km2",
                                 "green_to_built_km2", "green_gained_km2", "net_green_change_km2",
                                 "green_lost_km2_per_year"]]
        st.dataframe(show, hide_index=True, width="stretch")

    with st.expander("Technical findings (for the report)"):
        st.markdown(f"""
- **Growth is clustered, not spread out.** Moran's I = {hs["Moran's I, hexagon green->built km2"]:.2f}; {int(hs['Gi* hot hexagons (p<=0.05)'])} H3 hexagons are significant Gi* hotspots and hold {hs['Green->built km2 in hot hexagons (p<=0.05)']:.1f} km² of the conversion — the belt from Pettah and Chakkai to Sreevaraham and Kalippankulam.
- **Infill drives conversion.** FP-Growth on {int(rs.transactions):,} pixels: green land already surrounded by built-up in 2000 (`{top.rule}`) converted at {100 * top.confidence:.0f}%, {top.lift:.1f}× the city rate of {rs.base_conversion_rate_pct:.1f}%.
- **Roads matter.** {load_tab('road_distance').iloc[0].conversion_rate_pct:.1f}% of green within 100 m of a main road was converted, against {load_tab('road_distance').iloc[4].conversion_rate_pct:.1f}% at 1-2 km.
- **Converted land runs hotter.** Landsat LST rose {heat.loc['Green -> built-up', 'delta_c']:.1f} °C on converted pixels vs {heat.loc['Stable green', 'delta_c']:.1f} °C on stable green; greener wards are cooler (Spearman ρ = {case['Spearman rho, ward green % 2025 vs MODIS LST 2025']:.2f}).
- **Less green per person.** {case['Corporation green m2 per person, 2017 (WorldPop R2025A)']:.0f} → {case['Corporation green m2 per person, 2025 (WorldPop R2025A)']:.0f} m² per person (2017 → 2025); wards under 50 m²: {int(case['Wards below 50 m2 green per person, 2017 (WorldPop R2025A)'])} → {int(case['Wards below 50 m2 green per person, 2025 (WorldPop R2025A)'])}.
""")
    st.caption("Notes: green = tree cover + other vegetation (tree vs other is unreliable at 30 m). Earlier years are "
               "backdated from the 2025 map by change detection. The 2000 bare/sand class is suspect along the NW "
               "coast, which makes the green gain there look larger than it is. Population is compared only within "
               "one WorldPop release.")


def page_map():
    head("🗺️", "Map explorer", "Satellite land-cover maps for 2000, 2010, 2017 and 2025, with ward boundaries on top.")
    intro("Slide the year to watch red (built-up) spread into green. Tick *Green → built-up only* to see exactly "
          "where trees and fields were lost. Click a ward to select it.")
    opts = {"2000": "lc_2000", "2010": "lc_2010", "2017": "lc_2017", "2025": "lc_2025",
            "2025 (Sentinel-2, 10 m)": "lc_2025_s2"}
    c1, c2, c3 = st.columns([2, 1.2, 1.4])
    with c1:
        year = st.select_slider("Land-cover year", list(opts), value="2025")
    with c2:
        mode = st.radio("View", ["Single map", "Compare two years"], horizontal=True)
    with c3:
        metric = st.selectbox("Ward colour", ["(outline only)"] + list(WARD_METRICS),
                              format_func=lambda k: WARD_METRICS.get(k, k))
    metric = None if metric == "(outline only)" else metric
    with st.expander("Layers", expanded=True):
        lc = st.columns(4)
        show_lc = lc[0].checkbox("Land cover", True)
        show_change = lc[0].checkbox("Change 2000-2025", False)
        show_g2b = lc[1].checkbox("Green → built-up only", False)
        show_hex = lc[1].checkbox("Gi* hotspots", False)
        show_cl = lc[2].checkbox("DBSCAN clusters", False)
        show_roads = lc[2].checkbox("Main roads", True)
        show_low = lc[3].checkbox("Low-lying land", False)
        show_flood = lc[3].checkbox("2018 flood extent", False)
        opacity = st.slider("Overlay opacity", 0.2, 1.0, 0.8, 0.05)

    if mode == "Compare two years":
        year_b = st.select_slider("Compare with", list(opts), value="2000", key="year_b")
        dm = plugins.DualMap(location=CENTER, zoom_start=12, tiles=BASE_TILES)
        for mm, y in ((dm.m1, year_b), (dm.m2, year)):
            add_overlay(mm, opts[y], opacity=opacity)
            add_outline(mm)
        st.caption(f"Left: {year_b} · Right: {year}. The two maps pan and zoom together.")
        st.iframe(dm.get_root().render(), height=620)
        legend(LC_LEGEND)
        return

    m = base_map()
    if show_lc:
        add_overlay(m, opts[year], opacity=opacity)
    if show_change:
        add_overlay(m, "change", opacity=opacity)
    if show_g2b:
        add_overlay(m, "green_to_built", opacity=1.0)
    if show_low and "lowland" in OV:
        add_overlay(m, "lowland", opacity=opacity)
    if show_flood and "flood" in OV:
        add_overlay(m, "flood", opacity=1.0)
    if show_hex:
        hexes = load_geo("hexes")
        h = hexes[hexes.gi_class != "Not significant"]
        folium.GeoJson(h.to_json(), name="Gi* hotspots",
                       style_function=lambda f: {"fillColor": GI_COLORS[f["properties"]["gi_class"]],
                                                 "color": "#fff", "weight": 0.3, "fillOpacity": 0.75},
                       tooltip=folium.GeoJsonTooltip(["gi_class", "green_to_built_km2", "gi_z"],
                                                     ["Gi* class", "Green → built km²", "z"])).add_to(m)
    if show_cl:
        folium.GeoJson(load_geo("clusters").to_json(), name="DBSCAN clusters",
                       style_function=lambda f: {"color": "#67000d", "weight": 1.5, "fillColor": "#d7301f",
                                                 "fillOpacity": 0.25},
                       tooltip=folium.GeoJsonTooltip(["rank", "ward_name", "converted_km2"],
                                                     ["Cluster rank", "Main ward", "Converted km²"])).add_to(m)
    if show_roads:
        folium.GeoJson(load_geo("roads_main").to_json(), name="Main roads",
                       style_function=lambda f: {"color": "#333", "weight": 1.6 if f["properties"]["road_class"] == "highway" else 1},
                       tooltip=folium.GeoJsonTooltip(["name", "highway"], ["Road", "Type"])).add_to(m)
    ward_layer(m, metric, highlight=st.session_state.ward)
    add_outline(m)
    folium.LayerControl(collapsed=True).add_to(m)
    out = st_folium(m, height=620, use_container_width=True, key="explorer",
                    returned_objects=["last_active_drawing"])
    if show_lc:
        legend(LC_LEGEND)
    if show_change:
        legend(CHANGE_LEGEND)
    if show_hex:
        legend({k: v for k, v in GI_COLORS.items() if k != "Not significant"})
    w = clicked_ward(out)
    if w is not None and w != st.session_state.ward:
        st.session_state.ward = int(w)
        st.rerun()
    r = wards.set_index("ward_no").loc[st.session_state.ward]
    st.markdown(f"**Selected ward: {ward_label(st.session_state.ward)}** — green {r.green_pct_2000:.1f}% → "
                f"{r.green_pct_2025:.1f}% ({r.green_change_pp_2000_2025:+.1f} pp), priority rank "
                f"{int(r.priority_rank)} of {len(wards)}. Click a ward on the map to select it; open "
                f"*Ward drill-down* for details.")


def page_ward():
    head("🔎", "Ward drill-down", "One ward at a time: how green it was, how green it is, and how it compares with the city.")
    intro("Pick your ward to see how much green it had, how much it lost, and how it compares with the city "
          "(dashed lines). Rank 1 = the ward that needs action most.")
    nos = wards.sort_values("ward_no").ward_no.tolist()
    sel = st.selectbox("Ward", nos, index=nos.index(st.session_state.ward), format_func=ward_label)
    st.session_state.ward = int(sel)
    r = wards.set_index("ward_no").loc[sel]
    corp = load_tab("trend").pivot(index="year", columns="class", values="pct")
    c = st.columns(5)
    c[0].metric("Green cover 2025", f"{r.green_pct_2025:.1f}%", f"{r.green_change_pp_2000_2025:+.1f} pp since 2000")
    c[1].metric("Built-up 2025", f"{r.built_pct_2025:.1f}%", f"{r.built_pct_2025 - r.built_pct_2000:+.1f} pp",
                delta_color="inverse")
    c[2].metric("Green → built-up", f"{r.green_to_built_km2:.2f} km²", f"of {r.area_km2:.2f} km² ward", delta_color="off", delta_arrow="off")
    c[3].metric("Priority rank", f"{int(r.priority_rank)} / {len(wards)}",
                f"rank {int(r.rank_min)}-{int(r.rank_max)} across weightings", delta_color="off", delta_arrow="off")
    c[4].metric("Green per person 2025", f"{r.green_per_capita_2025:.0f} m²",
                f"{r.pop_2025_r25:,.0f} people", delta_color="off", delta_arrow="off")

    left, right = st.columns([1, 1])
    with left:
        d = pd.DataFrame({"year": YEARS * 2,
                          "pct": [r[f"green_pct_{y}"] for y in YEARS] + [r[f"built_pct_{y}"] for y in YEARS],
                          "series": ["Green (ward)"] * 4 + ["Built-up (ward)"] * 4})
        d2 = pd.DataFrame({"year": YEARS * 2,
                           "pct": list(corp[["Tree cover", "Other vegetation"]].sum(axis=1)) + list(corp["Built-up"]),
                           "series": ["Green (corporation)"] * 4 + ["Built-up (corporation)"] * 4})
        fig = px.line(pd.concat([d, d2]), x="year", y="pct", color="series", markers=True,
                      color_discrete_map={"Green (ward)": "#1a7837", "Built-up (ward)": "#d7301f",
                                          "Green (corporation)": "#9ccf9c", "Built-up (corporation)": "#f4a582"},
                      title="Green and built-up share (%)")
        fig.for_each_trace(lambda tr: tr.update(line_dash="dash") if "corporation" in tr.name else None)
        fig.update_xaxes(tickvals=YEARS, range=[1998, 2027], title=None)
        fig.update_yaxes(title=None, ticksuffix="%")
        plot(fig, 360)
        ind = pd.DataFrame({"indicator": list(INDICATORS.values()),
                            "ward": [r[f"n_{k}"] for k in INDICATORS],
                            "median ward": [wards[f"n_{k}"].median() for k in INDICATORS]})
        fig = px.bar(ind.melt(id_vars="indicator", var_name="who", value_name="score"), y="indicator", x="score",
                     color="who", barmode="group", orientation="h",
                     color_discrete_map={"ward": "#d7301f", "median ward": "#bbbbbb"},
                     title="Priority indicators (0 = lowest concern, 1 = highest)")
        fig.update_yaxes(title=None, autorange="reversed")
        plot(fig, 380)
    with right:
        geom = wards[wards.ward_no == sel]
        b = geom.total_bounds
        m = folium.Map(tiles=BASE_TILES, control_scale=True)
        folium.TileLayer("Esri.WorldImagery", name="Satellite (Esri)", show=False).add_to(m)
        add_overlay(m, "change", opacity=0.85)
        add_overlay(m, "lc_2025_s2", show=False, opacity=0.85)
        folium.GeoJson(wards[["geometry"]].to_json(), control=False,
                       style_function=lambda f: {"color": "#777", "weight": 0.6, "fill": False}).add_to(m)
        folium.GeoJson(geom[["geometry"]].to_json(), name="Ward", control=False,
                       style_function=lambda f: {"color": "#00bcd4", "weight": 3, "fill": False}).add_to(m)
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
        folium.LayerControl(collapsed=True).add_to(m)
        st_folium(m, height=420, use_container_width=True, key=f"ward_map_{sel}", returned_objects=[])
        legend(CHANGE_LEGEND, cols=3)
        facts = pd.DataFrame({
            "Indicator": ["Area", "Population 2025 (WorldPop R2025A)", "Density 2025", "Buildings (OSM)",
                          "Building footprint cover", "Road density", "Main-road density",
                          "Dry-season LST 2025 (MODIS)", "LST change 2003-2025",
                          "Low-lying green converted", "Area in Gi* hotspots", "Mapped 2018 flood"],
            "Value": [f"{r.area_km2:.2f} km²", f"{r.pop_2025_r25:,.0f}", f"{r.density_2025_per_km2:,.0f} /km²",
                      f"{r.buildings:,.0f}", f"{r.building_coverage_pct:.1f}%",
                      f"{r.road_density_km_per_km2:.1f} km/km²", f"{r.main_road_density_km_per_km2:.2f} km/km²",
                      f"{r.modis_lst_2025:.1f} °C", f"{r.modis_lst_change:+.2f} °C",
                      f"{r.lowland_loss_pct:.1f}% ({r.lowland_green_to_built_km2:.2f} km²)",
                      f"{r.hot_share_pct:.0f}%", f"{r.flood_2018_km2:.3f} km²"]})
        st.dataframe(facts, hide_index=True, width="stretch")
    row = wards[wards.ward_no == sel]
    d1, d2 = st.columns(2)
    d1.download_button("Download ward data (CSV)", row.drop(columns="geometry").to_csv(index=False),
                       f"ward_{sel}.csv", "text/csv")
    d2.download_button("Download ward boundary + data (GeoJSON)", row.to_json(), f"ward_{sel}.geojson",
                       "application/geo+json")


def page_hotspots():
    head("🔥", "Hotspots and drivers", "Where green loss clusters, and which conditions make land likely to be built over.")
    intro("**Hotspots:** red hexagons are areas where green loss is clustered far more than chance would give; "
          "blue = areas that kept their green. **Rules:** which conditions (near a road, surrounded by buildings, "
          "low-lying) make green land likely to be built over. *Lift 3* = three times the city average.")
    view = st.radio("Section", ["Hotspots (Gi* and DBSCAN)", "What drives conversion (FP-Growth)"], horizontal=True,
                    label_visibility="collapsed", key="hot_view")
    if view.startswith("Hotspots"):
        hs = load_tab("hotspot_summary").set_index("metric").value
        c = st.columns(4)
        moran = hs["Moran's I, hexagon green->built km2"]
        c[0].metric("🧲 Clustering score (Moran's I)", f"{moran:.2f}", "p = 0.001", delta_color="off",
                    delta_arrow="off", help="0 = loss scattered at random; closer to 1 = loss bunched together. "
                                            "p = 0.001: almost certainly not chance.")
        c[1].metric("🔥 Hotspot hexagons (Gi*, 95%)", f"{int(hs['Gi* hot hexagons (p<=0.05)'])}",
                    f"{int(hs['Gi* hot hexagons after FDR (q=0.05)'])} after FDR", delta_color="off", delta_arrow="off",
                    help="Hexagons where loss is much higher than in their neighbourhood. FDR is a stricter test "
                         "that allows for checking many hexagons at once.")
        c[2].metric("🏗️ Green lost inside hotspots", f"{hs['Green->built km2 in hot hexagons (p<=0.05)']:.1f} km²",
                    delta_color="off", delta_arrow="off",
                    help="Green land turned built-up inside the hotspot hexagons, 2000-2025.")
        c[3].metric("🧩 Building clusters (DBSCAN)", f"{int(hs['DBSCAN clusters'])}",
                    f"{hs['Green->built pixels in DBSCAN clusters (%)']:.0f}% of converted pixels", delta_color="off",
                    delta_arrow="off", help="Patches where many converted pixels sit close together "
                                            "(within 100 m), i.e. new neighbourhoods or estates.")
        f1, f2, f3 = st.columns([2, 1, 1])
        classes = f1.multiselect("Gi* classes", list(GI_COLORS), default=["Hot 99%", "Hot 95%", "Cold 99%", "Cold 95%"])
        min_km2 = f2.slider("Min. converted per hexagon (km²)", 0.0, 0.06, 0.0, 0.005)
        min_cl = f3.slider("Min. cluster size (km²)", 0.0, 0.3, 0.02, 0.01)
        hexes = load_geo("hexes")
        h = hexes[hexes.gi_class.isin(classes) & (hexes.green_to_built_km2 >= min_km2)]
        cl = load_geo("clusters")
        cl = cl[cl.converted_km2 >= min_cl]
        m = base_map()
        folium.GeoJson(h.to_json(), name="Gi* hexagons",
                       style_function=lambda f: {"fillColor": GI_COLORS[f["properties"]["gi_class"]], "color": "#fff",
                                                 "weight": 0.3, "fillOpacity": 0.8},
                       tooltip=folium.GeoJsonTooltip(["gi_class", "gi_z", "green_to_built_km2", "green_2000_pct",
                                                      "green_2025_pct"],
                                                     ["Gi* class", "z", "Green → built km²", "Green 2000 %",
                                                      "Green 2025 %"])).add_to(m)
        folium.GeoJson(cl.to_json(), name="DBSCAN clusters",
                       style_function=lambda f: {"color": "#3f007d", "weight": 2, "fillOpacity": 0.05},
                       tooltip=folium.GeoJsonTooltip(["rank", "ward_name", "converted_km2", "hull_km2"],
                                                     ["Rank", "Main ward", "Converted km²", "Hull km²"])).add_to(m)
        ward_layer(m, None, show=False)
        add_outline(m)
        folium.LayerControl(collapsed=True).add_to(m)
        st_folium(m, height=560, use_container_width=True, key="hotspot_map", returned_objects=[])
        legend({k: v for k, v in GI_COLORS.items() if k in classes} | {"DBSCAN cluster outline": "#3f007d"})
        st.caption(f"{len(h):,} hexagons and {len(cl)} clusters shown. Gi*: 6 nearest neighbours, 999 permutations; "
                   "DBSCAN: eps 100 m, at least 12 converted pixels.")
        st.dataframe(cl.drop(columns="geometry")[["rank", "ward_name", "converted_km2", "hull_km2", "wards"]]
                     .rename(columns={"ward_name": "main ward", "wards": "wards touched",
                                      "converted_km2": "green built over (km²)", "hull_km2": "cluster area (km²)"}),
                     hide_index=True, width="stretch", height=260)
        st.download_button("Download hotspots (GeoJSON)", h.to_json(), "gi_hotspots.geojson", "application/geo+json")

    else:
        rs = load_tab("rules_summary").iloc[0]
        st.markdown(f"FP-Growth (Spark MLlib) on **{int(rs.transactions):,}** green-2000 pixels, min. support "
                    f"{rs.min_support}, min. confidence {rs.min_confidence}: **{int(rs.frequent_itemsets):,}** frequent "
                    f"itemsets. City-wide, **{rs.base_conversion_rate_pct:.1f}%** of 2000 green became built-up; "
                    "*lift* says how many times that rate a rule reaches.")
        rules = load_tab("rules_conversion")
        cc = st.columns([2, 1, 1])
        q = cc[0].text_input("Filter rules containing (e.g. road=<100m, lowland, technopark)")
        min_lift = cc[1].slider("Min. lift", 1.0, float(rules.lift.max()), 1.5, 0.1)
        min_area = cc[2].slider("Min. converted area (km²)", 0.0, 2.0, 0.2, 0.05)
        r = rules[(rules.lift >= min_lift) & (rules.km2 >= min_area)]
        if q:
            r = r[r.rule.str.contains(q, regex=False)]
        st.dataframe(r.assign(confidence=(100 * r.confidence).round(1))
                     .rename(columns={"confidence": "converted %", "antecedent_km2": "green 2000 km² (if)",
                                      "km2": "converted km²"})[["rule", "green 2000 km² (if)", "converted km²",
                                                                 "converted %", "lift"]],
                     hide_index=True, width="stretch", height=330)
        st.download_button("Download rules (CSV)", r.to_csv(index=False), "rules_conversion.csv", "text/csv")
        st.subheader("One condition at a time")
        uni = load_tab("conversion_by_condition")
        attrs = sorted(uni.attribute.unique())
        a = st.selectbox("Condition", attrs, index=attrs.index("road") if "road" in attrs else 0)
        u = uni[uni.attribute == a]
        fig = px.bar(u, x="item", y="conversion_rate_pct", text="conversion_rate_pct",
                     hover_data=["green_2000_km2", "lift"], title=f"Share of 2000 green converted, by {a}")
        fig.add_hline(y=rs.base_conversion_rate_pct, line_dash="dash", annotation_text="city-wide")
        fig.update_traces(marker_color="#d7301f", texttemplate="%{text:.1f}%")
        fig.update_yaxes(title="converted %")
        plot(fig, 360)


# ---------------------------------------------------------------- spatial query

def pg_params():
    pw = os.environ.get("PGPASSWORD") or st.session_state.get("pg_pw") or None
    return {"host": os.environ.get("PGHOST", "localhost"), "port": int(os.environ.get("PGPORT", 5432)),
            "user": os.environ.get("PGUSER", "postgres"), "dbname": os.environ.get("TVM_DB", "tvm_green"),
            "password": pw, "connect_timeout": 3}


def pg_connect():
    import psycopg2
    return psycopg2.connect(**pg_params())


PG_SQL = """
WITH q AS (SELECT ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%(g)s), 4326), 32643) AS g)
SELECT count(*)                                         AS pixels,
       count(*) FILTER (WHERE p.lc_2000 IN (2, 3))      AS green_2000,
       count(*) FILTER (WHERE p.lc_2025 IN (2, 3))      AS green_2025,
       count(*) FILTER (WHERE p.lc_2000 = 1)            AS built_2000,
       count(*) FILTER (WHERE p.lc_2025 = 1)            AS built_2025,
       count(*) FILTER (WHERE p.change = 3)             AS green_to_built
FROM tvm.pixels30 p, q
WHERE ST_Intersects(p.geom, q.g);   -- GiST index scan on pixels30.geom
"""
PG_SQL_OSM = """
WITH q AS (SELECT ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%(g)s), 4326), 32643) AS g)
SELECT (SELECT count(*) FROM tvm.buildings b, q WHERE ST_Intersects(b.geom, q.g))                   AS buildings,
       (SELECT coalesce(sum(ST_Length(ST_Intersection(r.geom, q.g))), 0) / 1000
          FROM tvm.roads r, q WHERE ST_Intersects(r.geom, q.g))                                       AS road_km;
"""


def query_local(poly_utm):
    px_ = load_px()
    t = time.perf_counter()
    minx, miny, maxx, maxy = poly_utm.bounds
    box = px_[(px_.x >= minx) & (px_.x <= maxx) & (px_.y >= miny) & (px_.y <= maxy)]   # bounding-box filter first
    p = box[shapely.contains_xy(poly_utm, box.x.to_numpy(), box.y.to_numpy())]
    res = {"pixels": len(p), "green_2000": int(p.lc_2000.isin(GREEN).sum()),
           "green_2025": int(p.lc_2025.isin(GREEN).sum()), "built_2000": int((p.lc_2000 == 1).sum()),
           "built_2025": int((p.lc_2025 == 1).sum()), "green_to_built": int((p.change == 3).sum())}
    return res, (time.perf_counter() - t) * 1000


def query_pg(poly_wgs):
    t = time.perf_counter()
    g = json.dumps(mapping(poly_wgs))
    with pg_connect() as c, c.cursor() as cur:
        cur.execute(PG_SQL, {"g": g})
        cols = [d[0] for d in cur.description]
        res = dict(zip(cols, cur.fetchone()))
        ms = (time.perf_counter() - t) * 1000
        cur.execute(PG_SQL_OSM, {"g": g})
        res.update(dict(zip(["buildings", "road_km"], cur.fetchone())))
    c.close()
    return res, ms


def page_query():
    head("✏️", "Spatial query", "Draw any area on the map and get its green-loss numbers instantly.")
    intro("Use the toolbar on the left of the map: draw a shape, or drop a marker and pick a radius. The numbers "
          "below show how much green was inside in 2000 and 2025. With the *PostGIS* back end the query runs as "
          "spatial SQL on the 451 k-pixel table (GiST index) and also counts OSM buildings and road length.")
    with st.sidebar:
        st.subheader("PostGIS back end")
        if not os.environ.get("PGPASSWORD"):
            st.text_input("PostgreSQL password", type="password", key="pg_pw",
                          help="Kept only in this session's memory. Or set PGPASSWORD before starting the app.")
        try:
            c = pg_connect()
            c.close()
            pg_ok = True
            st.success(f"Connected to {pg_params()['dbname']}")
        except Exception as e:  # noqa: BLE001
            pg_ok = False
            st.info("Not connected — using the local pixel table.")
            st.caption(str(e).splitlines()[0][:150] if str(e) else "")
    c1, c2 = st.columns([1, 1])
    backend = c1.radio("Back end", ["PostGIS (live SQL)", "Local (GeoPandas / Shapely)"],
                       index=0 if pg_ok else 1, horizontal=True, disabled=not pg_ok)
    radius = c2.slider("Buffer radius for markers (m)", 100, 3000, 1000, 100)

    m = base_map(zoom=12)
    add_overlay(m, "change", opacity=0.7)
    ward_layer(m, None)
    add_outline(m)
    plugins.Draw(export=False, draw_options={"polyline": False, "circlemarker": False, "circle": False,
                                             "polygon": True, "rectangle": True, "marker": True},
                 edit_options={"edit": False}).add_to(m)
    out = st_folium(m, height=520, use_container_width=True, key="query_map",
                    returned_objects=["last_active_drawing", "all_drawings"])
    legend(CHANGE_LEGEND)
    drawings = (out or {}).get("all_drawings") or []
    if not drawings:
        st.info("Draw a shape on the map (toolbar on the left) to run a query.")
        return
    geom = shape(drawings[-1]["geometry"])
    if geom.geom_type == "Point":
        poly_utm = shp_transform(TO_UTM, geom).buffer(radius, 32)
        what = f"{radius:,} m buffer around ({geom.y:.4f}, {geom.x:.4f})"
    else:
        poly_utm = shp_transform(TO_UTM, geom)
        what = f"drawn {geom.geom_type.lower()}"
    poly_wgs = shp_transform(TO_WGS, poly_utm)
    if backend.startswith("PostGIS") and pg_ok:
        res, ms = query_pg(poly_wgs)
        used = "PostGIS"
    else:
        res, ms = query_local(poly_utm)
        used = "local"
    km = {k: res[k] * PX_KM2 for k in ("green_2000", "green_2025", "built_2000", "built_2025", "green_to_built")}
    area = poly_utm.area / 1e6
    st.subheader(f"Result for the {what} ({area:.2f} km²)")
    c = st.columns(5)
    c[0].metric("Green 2000", f"{km['green_2000']:.2f} km²")
    c[1].metric("Green 2025", f"{km['green_2025']:.2f} km²", f"{km['green_2025'] - km['green_2000']:+.2f} km²")
    c[2].metric("Built-up 2025", f"{km['built_2025']:.2f} km²", f"{km['built_2025'] - km['built_2000']:+.2f} km²",
                delta_color="inverse")
    c[3].metric("Green → built-up", f"{km['green_to_built']:.2f} km²",
                f"{100 * res['green_to_built'] / max(res['green_2000'], 1):.1f}% of 2000 green", delta_color="off", delta_arrow="off")
    c[4].metric("Query time", f"{ms:.0f} ms", f"{used}, {res['pixels']:,} pixels", delta_color="off", delta_arrow="off")
    if used == "PostGIS":
        st.caption(f"OSM inside the area: {res['buildings']:,} buildings, {res['road_km']:.1f} km of road.")
        with st.expander("SQL that ran"):
            st.code(PG_SQL.strip() + "\n\n" + PG_SQL_OSM.strip(), language="sql")
    fc = gpd.GeoDataFrame([{**{k: round(v, 4) for k, v in km.items()}, "area_km2": round(area, 4), "backend": used}],
                          geometry=[poly_wgs], crs=4326)
    st.download_button("Download query area + result (GeoJSON)", fc.to_json(), "query_result.geojson",
                       "application/geo+json")


# ---------------------------------------------------------------- decision support

def page_priority():
    head("🎯", "Where to act first", "A ward priority index you can tune to a policy goal.")
    intro("Darker wards need protection first. Pick a preset (e.g. *Climate risk*) and watch the ranking change; "
          "the table lists the top wards.")
    st.markdown("Where should the Corporation act first to protect remaining green space? Each indicator is scaled "
                "0-1 across the 101 wards (1 = most concern; 2nd-98th percentile clipped). Set the weights to match a "
                "policy goal; the ranking updates live.")
    preset = st.selectbox("Start from a preset", list(PRESETS))
    if st.session_state.get("_preset") != preset:
        for k, v in PRESETS[preset].items():
            st.session_state[f"w_{k}"] = v
        st.session_state._preset = preset
    w = {}
    with st.expander("⚖️ How much should each indicator count? (0 = ignore, 5 = most important)", expanded=True):
        cols = st.columns(4)
        for i, (k, label) in enumerate(INDICATORS.items()):
            w[k] = cols[i % 4].slider(label, 0, 5, key=f"w_{k}")
    if sum(w.values()) == 0:
        st.warning("Give at least one indicator a weight above 0.")
        return
    d = wards.copy()
    ws = pd.Series(w)
    d["index"] = (100 * sum(d[f"n_{k}"] * v for k, v in w.items()) / ws.sum()).round(1)
    d["rank"] = d["index"].rank(ascending=False, method="min").astype(int)
    d["change_vs_equal"] = d.priority_rank - d["rank"]
    top_n = st.slider("Highlight top N wards", 5, 40, 20)
    left, right = st.columns([1.1, 1])
    with left:
        m = base_map()
        cm = LinearColormap(["#ffffcc", "#fd8d3c", "#800026"], vmin=float(d["index"].min()),
                            vmax=float(d["index"].max()), caption="Priority index")
        cm.add_to(m)
        folium.GeoJson(
            d[["ward_no", "ward_name", "index", "rank", "geometry"]].to_json(), name="Priority index",
            style_function=lambda f: {"fillColor": cm(f["properties"]["index"]),
                                      "color": "#000" if f["properties"]["rank"] <= top_n else "#888",
                                      "weight": 2.2 if f["properties"]["rank"] <= top_n else 0.5,
                                      "fillOpacity": 0.75},
            tooltip=folium.GeoJsonTooltip(["ward_no", "ward_name", "index", "rank"],
                                          ["Ward no", "Ward", "Index", "Rank"])).add_to(m)
        add_outline(m)
        out = st_folium(m, height=560, use_container_width=True, key="priority_map",
                        returned_objects=["last_active_drawing"])
        wsel = clicked_ward(out)
        if wsel is not None:
            st.session_state.ward = int(wsel)
            st.caption(f"Selected {ward_label(int(wsel))} — open *Ward drill-down* for details.")
    with right:
        t = d.sort_values("rank").head(top_n)
        st.dataframe(t[["rank", "ward_name", "index", "change_vs_equal"] + list(INDICATORS)]
                     .rename(columns={"ward_name": "ward", "change_vs_equal": "Δ rank vs equal"} | INDICATORS),
                     hide_index=True, width="stretch", height=560)
    sens = load_tab("priority_sensitivity")
    st.caption("Robustness of the equal-weight ranking (Step 6): Spearman ρ with the other presets "
               + ", ".join(f"{r.scheme} {r.spearman_vs_equal:.2f} ({r.top20_overlap_with_equal}/20 shared)"
                           for r in sens.itertuples() if r.scheme != "equal") + ".")
    out_cols = ["ward_no", "ward_name", "index", "rank"] + list(INDICATORS) + [f"n_{k}" for k in INDICATORS]
    d1, d2 = st.columns(2)
    d1.download_button("Download ranking (CSV)", d.sort_values("rank")[out_cols].to_csv(index=False),
                       "ward_priority.csv", "text/csv")
    d2.download_button("Download ranking (GeoJSON)", d.sort_values("rank")[out_cols + ["geometry"]].to_json(),
                       "ward_priority.geojson", "application/geo+json")


# ---------------------------------------------------------------- big data engine

def page_bigdata():
    head("⚙️", "Big data engine", "How the numbers were computed: Spark, Sedona and PostGIS, and how fast each is.")
    intro("How the numbers were computed. Lower lines = faster. A spatial index makes the join about 60× faster "
          "than checking every point against every ward.")
    b = load_tab("bench_sedona")
    st.markdown("Every Sentinel-2 pixel (10 m) becomes a point and is joined to the 101 ward polygons, the core "
                "operation behind every ward statistic. Timed at growing input sizes (best of 3, local Spark, 12 "
                "threads).")
    log = st.toggle("Log scale", value=True)
    fig = px.line(b, x="points", y="seconds", color="label", markers=True, log_y=log,
                  title="Point-in-polygon join: runtime vs input size",
                  color_discrete_sequence=["#999999", "#2b8cbe", "#d7301f", "#fd8d3c", "#333333"])
    fig.update_xaxes(title="points joined")
    plot(fig, 430)
    big = b[b.points == b.points.max()].set_index("method")
    base = big.loc["spark_nested_loop", "seconds"]
    tbl = big[["label", "join_operator", "seconds"]].assign(speedup=(base / big.seconds).round(1))
    st.dataframe(tbl.reset_index(drop=True).rename(columns={
        "label": "method", "join_operator": "Spark join operator", "seconds": "seconds (4 M points)",
        "speedup": "× faster than nested loop"}), hide_index=True, width="stretch")
    st.caption("At 4 M points the whole dataset fits on one machine, so GeoPandas is fastest; Sedona's advantage is "
               "that its indexed joins scale linearly and can be spread over a cluster. The broadcast join wins "
               "here because the ward table is tiny.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spatial partitioning (Sedona)")
        parts = load_tab("partitions_sedona")
        if len(parts):
            st.dataframe(parts[["scheme", "partitions", "min_points", "max_points",
                                "mean_mbr_pct_of_extent"]].rename(columns={
                "partitions": "parts", "min_points": "min pts", "max_points": "max pts",
                "mean_mbr_pct_of_extent": "MBR %"}), hide_index=True, width="stretch")
        st.caption("All schemes split the same 4.04 M points. Min/max pts = fewest/most points in one partition "
                   "(load balance); "
                   "MBR % = how much of the extent each partition's bounding box covers (lower = more compact = "
                   "better pruning).")
    with c2:
        st.subheader("PostGIS: with and without GiST indexes")
        pg = load_tab("bench_postgis")
        if len(pg):
            pg = pg.assign(
                query=pg["query"].map({"pixel_in_ward_join": "Pixels → wards (451 k)",
                                       "buildings_in_ward_join": "Buildings → wards (203 k)",
                                       "range_query_1km": "Within 1 km of Technopark",
                                       "knn_nearest_main_road": "Nearest main road (KNN)",
                                       "partition_pruning": "Partition pruning"}).fillna(pg["query"]),
                mode=pg["mode"].map({"gist_index": "With GiST index", "no_index": "No index",
                                     "one_partition": "1 of 30 partitions", "all_partitions": "All 30 partitions"}),
                label=pg["label"].str.replace("->", "→"))
            fig = px.bar(pg, x="seconds", y="query", color="mode", barmode="group", orientation="h", log_x=True,
                         title="Query time in seconds (log scale, shorter = faster)",
                         color_discrete_map={"With GiST index": "#2b8cbe", "No index": "#999999",
                                             "1 of 30 partitions": "#1a7837", "All 30 partitions": "#cccccc"},
                         hover_data={"label": True, "plan_nodes": True, "query": False})
            fig.update_yaxes(title=None)
            fig.update_xaxes(title=None)
            plot(fig, 360)
    st.subheader("MapReduce view of the ward aggregation")
    st.code("map     : scan pixel points, find each point's ward (index join), emit key (ward_no, class)\n"
            "combine : HashAggregate partial_count inside each Spark task\n"
            "shuffle : Exchange hashpartitioning(ward_no, class)\n"
            "reduce  : HashAggregate count per key  →  km² per ward and class", language="text")
    acc = load_tab("classification_accuracy")
    if len(acc):
        with st.expander("Classification accuracy (Step 3)"):
            f1 = [c for c in acc.columns if c.startswith("f1_")]
            st.dataframe(acc[["composite", "n_train", "n_test", "overall_accuracy", "kappa", *f1]].rename(
                columns={"composite": "map", "n_train": "training pixels", "n_test": "test pixels",
                         "overall_accuracy": "overall accuracy", **{c: "F1 " + c[3:] for c in f1}}),
                hide_index=True, width="stretch")
            st.caption("F1 per class: 1 = perfect. Accuracy is measured against held-out reference pixels.")


{"Overview": page_overview, "Map explorer": page_map, "Ward drill-down": page_ward,
 "Hotspots & rules": page_hotspots, "Spatial query": page_query,
 "Priority (decision support)": page_priority, "Big data engine": page_bigdata}[page]()
