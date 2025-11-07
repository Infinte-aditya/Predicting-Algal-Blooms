import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Nal Sarovar Bloom Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data with caching."""
    historical = pd.read_csv('data/qgis_input/nal_sarovar_timeseries.csv', parse_dates=['date'])
    forecasts = pd.read_csv('outputs/predictions/bloom_forecast.csv', parse_dates=['date'])
    model_comparison = pd.read_csv('outputs/model_comparison.csv')
    
    return historical, forecasts, model_comparison

def create_timeline_plot(historical, forecasts, scenario):
    """Create interactive timeline plot combining historical and forecast data."""
    
    # Filter forecast by scenario
    forecast_scenario = forecasts[forecasts['scenario'] == scenario].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Historical data - NDVI
    fig.add_trace(go.Scatter(
        x=historical['date'],
        y=historical['ndvi'],
        name='Historical NDVI',
        mode='lines',
        line=dict(color='green', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>NDVI:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Historical blooms
    bloom_dates = historical[historical['bloom'] == 1]
    fig.add_trace(go.Scatter(
        x=bloom_dates['date'],
        y=bloom_dates['ndvi'],
        name='Historical Blooms',
        mode='markers',
        marker=dict(color='red', size=12, symbol='circle'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Bloom Detected</b><extra></extra>'
    ))
    
    # Forecast bloom probability
    fig.add_trace(go.Scatter(
        x=forecast_scenario['date'],
        y=forecast_scenario['bloom_probability'],
        name='Forecast Probability',
        mode='lines+markers',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=8),
        yaxis='y2',
        hovertemplate='<b>Date:</b> %{x}<br><b>Probability:</b> %{y:.1%}<extra></extra>'
    ))
    
    # Predicted high-risk periods
    high_risk = forecast_scenario[forecast_scenario['risk_level'] == 'High']
    fig.add_trace(go.Scatter(
        x=high_risk['date'],
        y=high_risk['bloom_probability'],
        name='High Risk Forecast',
        mode='markers',
        marker=dict(color='red', size=14, symbol='diamond'),
        yaxis='y2',
        hovertemplate='<b>Date:</b> %{x}<br><b>High Risk!</b><br><b>Probability:</b> %{y:.1%}<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title=f'Bloom Timeline - {scenario.capitalize()} Scenario',
        xaxis=dict(title='Date'),
        yaxis=dict(title='NDVI', side='left'),
        yaxis2=dict(title='Bloom Probability', side='right', overlaying='y', range=[0, 1]),
        hovermode='x unified',
        legend=dict(x=0, y=1, orientation='h'),
        height=500
    )
    
    return fig

def create_risk_heatmap(forecasts):
    """Create risk heatmap for all scenarios."""
    
    # Pivot data for heatmap
    pivot_data = forecasts.pivot_table(
        index='scenario',
        columns='date',
        values='bloom_probability',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Date", y="Scenario", color="Bloom Probability"),
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    
    fig.update_layout(
        title='Bloom Risk Heatmap - All Scenarios',
        height=300
    )
    
    return fig

def create_model_comparison_chart(model_comparison):
    """Create model comparison bar chart."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_comparison['model'],
        y=model_comparison['roc_auc'],
        name='ROC-AUC',
        marker_color='lightblue',
        text=model_comparison['roc_auc'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='ROC-AUC Score',
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🌊 Nal Sarovar Bloom Prediction System</p>', unsafe_allow_html=True)
    st.markdown('**Real-time monitoring and forecasting of algal blooms using machine learning**')
    st.markdown('---')
    
    # Load data
    try:
        historical, forecasts, model_comparison = load_data()
    except FileNotFoundError:
        st.error("⚠️ Data files not found. Please run the pipeline first:")
        st.code("python src/02_feature_engineering.py\npython src/03_train_models.py\npython src/04_predict_future.py")
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range selector
    st.sidebar.subheader("Historical Data Range")
    min_date = historical['date'].min().date()
    max_date = historical['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Scenario selector
    st.sidebar.subheader("Forecast Scenario")
    scenario = st.sidebar.selectbox(
        "Select Weather Scenario",
        options=['normal', 'wet', 'dry'],
        format_func=lambda x: x.capitalize()
    )
    
    # Threshold selector
    st.sidebar.subheader("Risk Threshold")
    risk_threshold = st.sidebar.slider(
        "Bloom Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability above which a bloom is predicted"
    )
    
    # Main content
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_blooms = historical['bloom'].sum()
        st.metric("Historical Blooms", f"{total_blooms}")
    
    with col2:
        forecast_scenario = forecasts[forecasts['scenario'] == scenario]
        predicted_blooms = (forecast_scenario['bloom_probability'] > risk_threshold).sum()
        st.metric(f"Predicted Blooms ({scenario.capitalize()})", f"{predicted_blooms}")
    
    with col3:
        avg_risk = forecast_scenario['bloom_probability'].mean()
        st.metric("Average Forecast Risk", f"{avg_risk:.1%}")
    
    with col4:
        high_risk_count = (forecast_scenario['risk_level'] == 'High').sum()
        st.metric("High Risk Periods", f"{high_risk_count}")
    
    st.markdown("---")
    
    # Timeline Plot
    st.subheader("📈 Historical Data & Forecast Timeline")
    timeline_fig = create_timeline_plot(historical, forecasts, scenario)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Risk Heatmap
        st.subheader("🔥 Risk Heatmap - All Scenarios")
        heatmap_fig = create_risk_heatmap(forecasts)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with col_right:
        # Model Performance
        st.subheader("🎯 Model Performance")
        model_fig = create_model_comparison_chart(model_comparison)
        st.plotly_chart(model_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Forecast Details Table
    st.subheader(f"📋 Detailed Forecast - {scenario.capitalize()} Scenario")
    
    forecast_display = forecast_scenario[['date', 'bloom_probability', 'risk_level']].copy()
    forecast_display['date'] = forecast_display['date'].dt.date
    forecast_display['bloom_probability'] = forecast_display['bloom_probability'].apply(lambda x: f"{x:.1%}")
    forecast_display.columns = ['Date', 'Bloom Probability', 'Risk Level']
    
    # Color code risk levels
    def color_risk(val):
        if val == 'High':
            return 'background-color: #ffcccc'
        elif val == 'Medium':
            return 'background-color: #fff4cc'
        else:
            return 'background-color: #ccffcc'
    
    styled_table = forecast_display.style.applymap(color_risk, subset=['Risk Level'])
    st.dataframe(styled_table, use_container_width=True)
    
    # Download button
    csv = forecast_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"bloom_forecast_{scenario}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Nal Sarovar Bloom Prediction System | Powered by Machine Learning</p>
        <p>Data Sources: NDVI (Satellite), Temperature (Ahmedabad), Rainfall (Surendranagar)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()