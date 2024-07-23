import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

if selected_type == 'Both':
    consumption_data = filtered_data[filtered_data['Type'] == 'Consumption'].groupby('periodid')['value'].sum().reset_index()
    service_data = filtered_data[filtered_data['Type'] == 'Service'].groupby('periodid')['value'].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=consumption_data['periodid'], y=consumption_data['value'],
                             mode='lines+markers', name='Consumption', line=dict(color='firebrick')))
    fig.add_trace(go.Scatter(x=service_data['periodid'], y=service_data['value'],
                             mode='lines+markers', name='Service', line=dict(color='blue')))
else:
    filtered_data = filtered_data[filtered_data['Type'] == selected_type]
    aggregated_data = filtered_data.groupby('periodid')['value'].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aggregated_data['periodid'], y=aggregated_data['value'],
                             mode='lines+markers', name=selected_type))

fig.update_layout(title=f'Trend of {product} in {county}', xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig)

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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aggregated_data['periodid'], y=aggregated_data['value'], mode='lines+markers', name='Observed'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int['lower'], fill=None, mode='lines', line_color='gray', name='Lower CI'))
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int['upper'], fill='tonexty', mode='lines', line_color='gray', name='Upper CI'))
    fig.update_layout(title=f'Forecast of {product} in {county}', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

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
