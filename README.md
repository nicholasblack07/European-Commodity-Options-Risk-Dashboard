# Black–Scholes Options Pricing & Risk Dashboard

Interactive Streamlit dashboard for theoretical European call/put prices, five Greeks and three sensitivity charts.

## Setup

Install Python 3.10+ and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features

- Editable spot, strike, maturity, interest rate and volatility
- European call and put prices with zero dividends
- Delta, Gamma, Vega, Theta and Rho (Vega/Rho per 1 percentage point, Theta per calendar day)
- Interactive Plotly sensitivity curves for spot, volatility and maturity

## Assumptions and limitations

Constant volatility and risk-free rate; European exercise; no dividends or transaction costs. This is an educational model, not trading advice.

## Suggested tests

- Call-put parity: call - put = S - K exp(-rT)
- Call price should increase with underlying price and volatility
- Compare results with an independent reference implementation
