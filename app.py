import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def black_scholes(S, X, T, r, sigma, option_type='call'):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

# Streamlit App
st.title('Black-Scholes Option Pricing Model')

st.sidebar.header('Input Parameters')
S = st.sidebar.number_input('Current Stock Price (S)', min_value=0.0, value=100.0)
X = st.sidebar.number_input('Strike Price (X)', min_value=0.0, value=100.0)
T = st.sidebar.number_input('Time to Maturity (T, in years)', min_value=0.0, value=1.0)
r = st.sidebar.number_input('Risk-free Interest Rate (r)', min_value=0.0, max_value=1.0, value=0.05)
sigma = st.sidebar.number_input('Volatility (σ)', min_value=0.0, max_value=1.0, value=0.2)
option_type = st.sidebar.selectbox('Option Type', ('call', 'put'))

price = black_scholes(S, X, T, r, sigma, option_type)

st.write(f'The {option_type} option price is: ${price:.2f}')

# Plotting the option price sensitivity
S_range = np.linspace(S * 0.5, S * 1.5, 100)
price_range = [black_scholes(s, X, T, r, sigma, option_type) for s in S_range]

fig = go.Figure()
fig.add_trace(go.Scatter(x=S_range, y=price_range, mode='lines', name='Option Price'))
fig.update_layout(title='Option Price Sensitivity to Stock Price',
                  xaxis_title='Stock Price',
                  yaxis_title='Option Price')

st.plotly_chart(fig)

# Create and display heatmap
strike_prices = np.linspace(S * 0.5, S * 1.5, 50)
volatilities = np.linspace(0.1, 1.0, 50)
X_grid, sigma_grid = np.meshgrid(strike_prices, volatilities)
price_grid = np.zeros_like(X_grid)

for i in range(len(strike_prices)):
    for j in range(len(volatilities)):
        price_grid[j, i] = black_scholes(S, X_grid[j, i], T, r, sigma_grid[j, i], option_type='call')

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(price_grid, cmap='viridis', cbar_kws={'label': 'Call Option Price'})

# Customize ticks
ax = plt.gca()
xticks = np.linspace(0, len(strike_prices) - 1, 5).astype(int)  # Convert to integers
yticks = np.linspace(0, len(volatilities) - 1, 5).astype(int)  # Convert to integers
ax.set_xticks(xticks)
ax.set_xticklabels(np.round(strike_prices[xticks], 2))  # No need for .astype(float)
ax.set_yticks(yticks)
ax.set_yticklabels(np.round(volatilities[yticks], 2))  # No need for .astype(float)

plt.title('Heatmap of Call Option Prices')
plt.xlabel('Strike Price')
plt.ylabel('Volatility')
st.pyplot(plt)

# Attribution text
st.markdown("---")  # Adds a horizontal line for separation
st.markdown("Made by [Patrick Kucharski](https://www.seek.com.au/profile/patrick-kucharski-CdXCDGM3Vg)")