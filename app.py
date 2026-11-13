import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Black–Scholes Risk Dashboard", layout="wide")
st.title("Black–Scholes Options Pricing & Risk Dashboard")
st.caption("European options on a non-dividend-paying asset; constant volatility and risk-free rate.")

def prices(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put, d1, d2

with st.sidebar:
    st.header("Market inputs")
    S = st.slider("Underlying price", 10.0, 250.0, 100.0, 1.0)
    K = st.slider("Strike price", 10.0, 250.0, 100.0, 1.0)
    T = st.slider("Years to expiry", 0.05, 3.0, 1.0, 0.05)
    r_pct = st.slider("Risk-free rate (%)", 0.0, 15.0, 5.0, 0.25)
    sigma_pct = st.slider("Volatility (%)", 1.0, 100.0, 20.0, 1.0)
    option_type = st.radio("Greeks for", ["Call", "Put"], horizontal=True)
r = r_pct / 100
sigma = sigma_pct / 100
call, put, d1, d2 = prices(S, K, T, r, sigma)
a, b = st.columns(2)
a.metric("Call price", f"{call:.4f}")
b.metric("Put price", f"{put:.4f}")

pdf = norm.pdf(d1)
if option_type == "Call":
    delta = norm.cdf(d1)
    theta = -(S * pdf * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
else:
    delta = norm.cdf(d1) - 1
    theta = -(S * pdf * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
gamma = pdf / (S * sigma * np.sqrt(T))
vega = S * pdf * np.sqrt(T)
st.subheader(f"{option_type} option Greeks")
cols = st.columns(5)
for col, (name, value, detail) in zip(cols, [
    ("Delta", delta, "per 1 unit of underlying"),
    ("Gamma", gamma, "delta change per 1 unit"),
    ("Vega", vega / 100, "per 1 vol percentage point"),
    ("Theta", theta / 365, "per calendar day"),
    ("Rho", rho / 100, "per 1 rate percentage point"),
]):
    col.metric(name, f"{value:.4f}", help=detail)

st.subheader("Sensitivity analysis")
figures = []
spot_grid = np.linspace(max(1, S * 0.5), S * 1.5, 100)
vol_grid = np.linspace(0.01, min(1.5, max(0.6, sigma * 2)), 100)
time_grid = np.linspace(0.02, max(1.5, T * 1.5), 100)
for title, x, cp, pp, x_label in [
    ("Price vs underlying", spot_grid, *prices(spot_grid, K, T, r, sigma)[:2], "Underlying price"),
    ("Price vs volatility", vol_grid * 100, *prices(S, K, T, r, vol_grid)[:2], "Volatility (%)"),
    ("Price vs time to expiry", time_grid, *prices(S, K, time_grid, r, sigma)[:2], "Years to expiry"),
]:
    fig = go.Figure()
    fig.add_scatter(x=x, y=cp, name="Call", mode="lines")
    fig.add_scatter(x=x, y=pp, name="Put", mode="lines")
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Option price", height=350, margin=dict(l=20, r=20, t=45, b=20))
    figures.append(fig)
left, right = st.columns(2)
left.plotly_chart(figures[0], width="stretch")
right.plotly_chart(figures[1], width="stretch")
st.plotly_chart(figures[2], width="stretch")
st.info("Educational model only. Real options may be American-style, dividend-paying, and subject to volatility smiles and transaction costs.")
# Volatility heatmap
st.subheader("Volatility Heatmap")

st.write(
    "Explore how option prices change with the "
    "underlying asset price and implied volatility."
)

# Generate different underlying prices and volatilities
spot_range = np.linspace(S * 0.7, S * 1.3, 25)
vol_range = np.linspace(0.05, 0.80, 25)

# Calculate option prices for every combination
spot_mesh, vol_mesh = np.meshgrid(spot_range, vol_range)

call_heatmap, put_heatmap, _, _ = prices(
    spot_mesh, K, T, r, vol_mesh
)

# Allow users to switch between call and put prices
heatmap_type = st.radio(
    "Heatmap option type",
    ["Call", "Put"],
    horizontal=True
)

heatmap_values = (
    call_heatmap if heatmap_type == "Call" else put_heatmap
)

# Create interactive heatmap
heatmap_fig = go.Figure(
    data=go.Heatmap(
        x=spot_range,
        y=vol_range * 100,
        z=heatmap_values,
        colorscale="Viridis",
        colorbar=dict(title="Option price"),
        hovertemplate=(
            "Underlying: %{x:.2f}<br>"
            "Volatility: %{y:.1f}%<br>"
            "Option price: %{z:.2f}<extra></extra>"
        )
    )
)

heatmap_fig.update_layout(
    title=f"{heatmap_type} Option Price Heatmap",
    xaxis_title="Underlying Asset Price",
    yaxis_title="Volatility (%)",
    height=550
)

st.plotly_chart(heatmap_fig, width="stretch")
# ==========================================
# DELTA AND GAMMA RISK HEATMAPS
# ==========================================

st.header("Delta & Gamma Risk Analysis")

st.write(
    "Analyse how option sensitivities change "
    "across underlying prices and volatility levels."
)

# Create grids of underlying prices and volatility
risk_spot_range = np.linspace(S * 0.7, S * 1.3, 40)
risk_vol_range = np.linspace(0.05, 0.80, 40)

risk_spot_mesh, risk_vol_mesh = np.meshgrid(
    risk_spot_range, risk_vol_range
)

# Calculate Black-Scholes d1
risk_d1 = (
    np.log(risk_spot_mesh / K)
    + (r + 0.5 * risk_vol_mesh**2) * T
) / (risk_vol_mesh * np.sqrt(T))

# Delta calculations
call_delta = norm.cdf(risk_d1)
put_delta = call_delta - 1

# Gamma calculation (same for calls and puts)
gamma_values = (
    norm.pdf(risk_d1)
    / (risk_spot_mesh * risk_vol_mesh * np.sqrt(T))
)

# Allow user to choose call or put
risk_option_type = st.radio(
    "Risk heatmap option type",
    ["Call", "Put"],
    horizontal=True,
    key="risk_option_selector"
)

delta_values = (
    call_delta if risk_option_type == "Call" else put_delta
)

# Delta heatmap
delta_fig = go.Figure(
    data=go.Heatmap(
        x=risk_spot_range,
        y=risk_vol_range * 100,
        z=delta_values,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        zmid=0,
        colorbar=dict(title="Delta"),
        hovertemplate=(
            "Underlying: %{x:.2f}<br>"
            "Volatility: %{y:.1f}%<br>"
            "Delta: %{z:.4f}<extra></extra>"
        )
    )
)

delta_fig.update_layout(
    title=f"{risk_option_type} Option Delta",
    xaxis_title="Underlying Asset Price",
    yaxis_title="Volatility (%)",
    height=450
)

# Gamma heatmap
gamma_fig = go.Figure(
    data=go.Heatmap(
        x=risk_spot_range,
        y=risk_vol_range * 100,
        z=gamma_values,
        colorscale="YlOrRd",
        colorbar=dict(title="Gamma"),
        hovertemplate=(
            "Underlying: %{x:.2f}<br>"
            "Volatility: %{y:.1f}%<br>"
            "Gamma: %{z:.6f}<extra></extra>"
        )
    )
)

gamma_fig.update_layout(
    title="Option Gamma Exposure",
    xaxis_title="Underlying Asset Price",
    yaxis_title="Volatility (%)",
    height=450
)

# Display charts
delta_col, gamma_col = st.columns(2)

with delta_col:
    st.plotly_chart(delta_fig, width="stretch")

with gamma_col:
    st.plotly_chart(gamma_fig, width="stretch")

st.info(
    "Delta measures directional exposure. "
    "Gamma measures how quickly Delta changes. "
    "Both are important for options hedging and risk management."
)
# ==========================================
# BLACK-76 COMMODITY FUTURES OPTIONS MODEL
# ==========================================

st.divider()
st.header("Black-76 Commodity Futures Options Model")

st.write(
    "Price European-style options on commodity futures "
    "using the Black-76 model."
)

# Illustrative inputs, not live market prices

# European commodity futures and benchmarks
# Prices and volatilities are illustrative, not live

commodity_presets = {
    # Base Metals
    "LME Copper | USD/t": (10000.0, 25.0),
    "LME Aluminium | USD/t": (2600.0, 22.0),
    "LME Zinc | USD/t": (2900.0, 27.0),
    "LME Nickel | USD/t": (17000.0, 40.0),
    "LME Lead | USD/t": (2100.0, 23.0),
    "LME Tin | USD/t": (32000.0, 35.0),

    # Precious Metals
    "Gold | USD/oz": (3000.0, 20.0),
    "Silver | USD/oz": (35.0, 32.0),
    "Platinum | USD/oz": (1100.0, 30.0),
    "Palladium | USD/oz": (1000.0, 38.0),

    # Ferrous Metals
    "LME Steel Scrap Turkey | USD/t": (350.0, 30.0),
    "LME Steel Rebar Turkey | USD/t": (550.0, 28.0),
    "European HRC | EUR/t": (650.0, 28.0),
    "Iron Ore 62% Fe | USD/dmt": (105.0, 35.0),

    # Battery and Critical Metals
    "Lithium Hydroxide | USD/t": (15000.0, 55.0),
    "Cobalt | USD/t": (35000.0, 40.0),
    "Lithium Carbonate | USD/t": (14000.0, 55.0),
    "Spodumene | USD/t": (1100.0, 50.0),
    "Cobalt Hydroxide | USD/t": (16000.0, 45.0),

    # Fertilisers
    "Urea FOB Egypt | USD/t": (450.0, 35.0),
    "Urea FOB Middle East | USD/t": (420.0, 35.0),
    "Urea FOB Baltic | USD/t": (400.0, 38.0),
    "Urea FCA France | EUR/t": (500.0, 32.0),
    "Ammonia CFR NW Europe | USD/t": (650.0, 40.0),
    "UAN 30 FCA Rouen | EUR/t": (350.0, 30.0),
    "DAP FCA Benelux | USD/t": (650.0, 30.0),
    "CAN 27 Germany | EUR/t": (350.0, 28.0),
    "Potash MOP Europe | EUR/t": (350.0, 30.0),

    # Energy
    "ICE Brent Crude | USD/bbl": (75.0, 32.0),
    "ICE Gasoil | USD/t": (700.0, 35.0),
    "Dutch TTF Gas | EUR/MWh": (35.0, 55.0),
    "UK NBP Gas | p/therm": (90.0, 55.0),
    "German Power | EUR/MWh": (90.0, 60.0),
    "French Power | EUR/MWh": (75.0, 65.0),
    "EU Carbon Allowances | EUR/tCO2": (75.0, 35.0),

    # Agriculture
    "Euronext Milling Wheat | EUR/t": (230.0, 25.0),
    "Euronext Rapeseed | EUR/t": (480.0, 28.0),
    "Euronext Corn | EUR/t": (220.0, 27.0),
    "ICE UK Feed Wheat | GBP/t": (200.0, 25.0),

    # Custom
    "Custom Commodity": (100.0, 25.0)
}

commodity_categories = {
    "Base Metals": list(commodity_presets.keys())[0:6],
    "Precious Metals": list(commodity_presets.keys())[6:10],
    "Ferrous Metals": list(commodity_presets.keys())[10:14],
    "Battery & Critical Metals": list(commodity_presets.keys())[14:19],
    "Fertilisers": list(commodity_presets.keys())[19:28],
    "Energy": list(commodity_presets.keys())[28:35],
    "Agriculture": list(commodity_presets.keys())[35:39],
    "Custom": list(commodity_presets.keys())[39:]
}

selected_category = st.selectbox(
    "Select Commodity Sector",
    list(commodity_categories.keys())
)

commodity = st.selectbox(
    "Select Commodity / Benchmark",
    commodity_categories[selected_category]
)

default_futures, default_vol = commodity_presets[commodity]

# Futures market inputs
b1, b2, b3 = st.columns(3)

with b1:
    F = st.number_input(
        "Futures price",
        min_value=0.01,
        value=default_futures,
        key="black76_futures"
    )

with b2:
    K76 = st.number_input(
        "Strike price",
        min_value=0.01,
        value=default_futures,
        key="black76_strike"
    )

with b3:
    sigma76_pct = st.slider(
        "Futures volatility (%)",
        1.0, 100.0, default_vol,
        key="black76_vol"
    )

b4, b5 = st.columns(2)

with b4:
    T76 = st.slider(
        "Years to option expiry",
        0.05, 3.0, 1.0,
        key="black76_expiry"
    )

with b5:
    r76_pct = st.slider(
        "Risk-free interest rate (%)",
        0.0, 15.0, 5.0,
        key="black76_rate"
    )

sigma76 = sigma76_pct / 100
r76 = r76_pct / 100


# Black-76 pricing function
def black76(F, K, T, r, sigma):
    F = np.asarray(F, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    d1 = (
        np.log(F / K) + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    discount = np.exp(-r * T)

    call = discount * (
        F * norm.cdf(d1) - K * norm.cdf(d2)
    )

    put = discount * (
        K * norm.cdf(-d2) - F * norm.cdf(-d1)
    )

    return call, put, d1, d2


# Calculate futures option prices
call76, put76, d1_76, d2_76 = black76(
    F, K76, T76, r76, sigma76
)

st.subheader("Futures Option Valuation")

c1, c2 = st.columns(2)

c1.metric("Black-76 Call Price", f"{call76:,.4f}")
c2.metric("Black-76 Put Price", f"{put76:,.4f}")

st.caption(
    "Prices are quoted per unit of the commodity. "
    "Multiply by the relevant contract size for "
    "an illustrative contract premium."
)


# ==========================================
# BLACK-76 GREEKS
# ==========================================

discount76 = np.exp(-r76 * T76)
pdf76 = norm.pdf(d1_76)

call_delta76 = discount76 * norm.cdf(d1_76)
put_delta76 = discount76 * (norm.cdf(d1_76) - 1)

gamma76 = (
    discount76 * pdf76
    / (F * sigma76 * np.sqrt(T76))
)

vega76 = (
    discount76 * F * pdf76 * np.sqrt(T76)
)

# Theta: calendar-time decay with fixed futures price
theta_call76 = (
    r76 * call76
    - discount76 * F * pdf76 * sigma76
    / (2 * np.sqrt(T76))
)

theta_put76 = (
    r76 * put76
    - discount76 * F * pdf76 * sigma76
    / (2 * np.sqrt(T76))
)

# Rho: discount-rate sensitivity holding F constant
rho_call76 = -T76 * call76
rho_put76 = -T76 * put76

greek_type76 = st.radio(
    "Greeks for",
    ["Call", "Put"],
    horizontal=True,
    key="black76_greek_type"
)

selected_delta76 = (
    call_delta76 if greek_type76 == "Call"
    else put_delta76
)

selected_theta76 = (
    theta_call76 if greek_type76 == "Call"
    else theta_put76
)

selected_rho76 = (
    rho_call76 if greek_type76 == "Call"
    else rho_put76
)

st.subheader("Black-76 Risk Sensitivities")

greek_columns76 = st.columns(5)

greek_data76 = [
    ("Delta", selected_delta76),
    ("Gamma", gamma76),
    ("Vega / 1% vol", vega76 / 100),
    ("Theta / day", selected_theta76 / 365),
    ("Rho / 1% rate", selected_rho76 / 100)
]

for col, (name, value) in zip(
    greek_columns76, greek_data76
):
    col.metric(name, f"{value:.4f}")


# ==========================================
# FUTURES PRICE SENSITIVITY CHART
# ==========================================

st.subheader("Futures Price Sensitivity")

futures_range = np.linspace(
    F * 0.7, F * 1.3, 100
)

call_curve76, put_curve76, _, _ = black76(
    futures_range, K76, T76, r76, sigma76
)

fig_futures76 = go.Figure()

fig_futures76.add_trace(
    go.Scatter(
        x=futures_range,
        y=call_curve76,
        name="Call",
        mode="lines"
    )
)

fig_futures76.add_trace(
    go.Scatter(
        x=futures_range,
        y=put_curve76,
        name="Put",
        mode="lines"
    )
)

fig_futures76.update_layout(
    title="Option Price vs Futures Price",
    xaxis_title="Futures Price",
    yaxis_title="Option Premium per Unit",
    height=450
)

st.plotly_chart(fig_futures76, width="stretch")


# ==========================================
# FUTURES VOLATILITY HEATMAP
# ==========================================

st.subheader("Commodity Futures Volatility Heatmap")

heatmap_futures = np.linspace(
    F * 0.7, F * 1.3, 40
)

heatmap_vols = np.linspace(
    0.05, 1.0, 40
)

F_mesh76, vol_mesh76 = np.meshgrid(
    heatmap_futures, heatmap_vols
)

call_grid76, put_grid76, _, _ = black76(
    F_mesh76, K76, T76, r76, vol_mesh76
)

heatmap_choice76 = st.radio(
    "Heatmap option type",
    ["Call", "Put"],
    horizontal=True,
    key="black76_heatmap_choice"
)

heatmap_z76 = (
    call_grid76 if heatmap_choice76 == "Call"
    else put_grid76
)

fig_heatmap76 = go.Figure(
    data=go.Heatmap(
        x=heatmap_futures,
        y=heatmap_vols * 100,
        z=heatmap_z76,
        colorscale="Viridis",
        colorbar=dict(title="Premium"),
        hovertemplate=(
            "Futures price: %{x:.2f}<br>"
            "Volatility: %{y:.1f}%<br>"
            "Premium: %{z:.4f}<extra></extra>"
        )
    )
)

fig_heatmap76.update_layout(
    title=f"{commodity} — {heatmap_choice76} Premium",
    xaxis_title="Futures Price",
    yaxis_title="Volatility (%)",
    height=500
)

st.plotly_chart(fig_heatmap76, width="stretch")

st.info(
    "Black-76 assumes lognormal futures prices, "
    "constant volatility and interest rates, and "
    "European-style exercise. Commodity futures "
    "options may require different models when "
    "prices can become negative or when exercise "
    "and settlement conventions differ."
)
