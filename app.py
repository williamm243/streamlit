import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
def load_data():
    file_path = 'combined_product_data (1).csv'
    data = pd.read_csv(file_path)
    data['periodid'] = pd.to_datetime(data['periodid'], format='%Y%m')
    return data

data = load_data()

# Streamlit app
st.title('Product Consumption Trend and Forecast')

# Sidebar filters
product = st.sidebar.selectbox('Select Product', data['common_name'].unique())
county = st.sidebar.selectbox('Select County', data['organisationunitname'].unique())
type_options = ['Consumption', 'Service', 'Both']
selected_type = st.sidebar.selectbox('Select Type', type_options)

# Filter data based on selections
filtered_data = data[(data['common_name'] == product) & 
                     (data['organisationunitname'] == county)]

# Display the trend
st.subheader(f'Trend for {product} in {county}')

fig, ax = plt.subplots(figsize=(10, 5))

if selected_type == 'Both':
    consumption_data = filtered_data[filtered_data['Type'] == 'Consumption'].groupby('periodid')['value'].sum().reset_index()
    service_data = filtered_data[filtered_data['Type'] == 'Service'].groupby('periodid')['value'].sum().reset_index()

    ax.plot(consumption_data['periodid'], consumption_data['value'], marker='o', linestyle='-', label='Consumption', color='firebrick')
    ax.plot(service_data['periodid'], service_data['value'], marker='x', linestyle='-', label='Service', color='blue')
else:
    filtered_data = filtered_data[filtered_data['Type'] == selected_type]
    aggregated_data = filtered_data.groupby('periodid')['value'].sum().reset_index()

    ax.plot(aggregated_data['periodid'], aggregated_data['value'], marker='o', linestyle='-', label=selected_type)

ax.set_title(f'Trend of {product} in {county}')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Forecasting
st.subheader(f'Forecast for {product} in {county}')

if selected_type != 'Both':
    # Fit the SARIMAX model
    model = SARIMAX(aggregated_data['value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False, maxiter=1000, method='nm')  # Increased iterations and changed optimizer

    # Forecast for the next 12 periods
    forecast = model_fit.get_forecast(steps=12)
    forecast_index = pd.date_range(start=aggregated_data['periodid'].iloc[-1], periods=13, freq='M')[1:]
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    conf_int.columns = ['lower', 'upper']

    # Plot the forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(aggregated_data['periodid'], aggregated_data['value'], marker='o', linestyle='-', label='Observed')
    ax.plot(forecast_index, forecast_values, marker='x', linestyle='-', label='Forecast')
    ax.fill_between(forecast_index, conf_int['lower'], conf_int['upper'], color='gray', alpha=0.2, label='Confidence Interval')
    ax.set_title(f'Forecast of {product} in {county}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Display forecast table
    st.subheader('Forecast Table')
    forecast_table = pd.DataFrame({
        'Date': forecast_index,
        'Forecast': forecast_values,
        'Lower CI': conf_int['lower'],
        'Upper CI': conf_int['upper']
    })
    st.write(forecast_table)
else:
    st.write("Forecasting is only available when a single type is selected (Consumption or Service).")
