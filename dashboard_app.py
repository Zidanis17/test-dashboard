import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import folium
from streamlit_folium import folium_static
import numpy as np
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mobile Network Analytics Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.4);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .success-alert {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(40, 167, 69, 0.05));
        border-left-color: #28a745;
        color: #155724;
    }
    .warning-alert {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
        border-left-color: #ffc107;
        color: #856404;
    }
    .danger-alert {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1), rgba(220, 53, 69, 0.05));
        border-left-color: #dc3545;
        color: #721c24;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the network performance data"""
    try:
        df = pd.read_csv('dashboard_ready_with_predictions.csv')
        
        # Create timestamp from existing columns
        if 'month' in df.columns and 'day' in df.columns and 'hour' in df.columns:
            # Infer year: December = 2024, others = 2025
            df['year'] = 2025
            df.loc[df['month'] == 12, 'year'] = 2024
            
            # Create timestamp
            df['Timestamp'] = pd.to_datetime(
                df['year'].astype(str) + '-' + 
                df['month'].astype(str) + '-' + 
                df['day'].astype(str) + ' ' + 
                df['hour'].astype(str) + ':00:00',
                errors='coerce'
            )
        else:
            # Fallback timestamp
            base_date = datetime(2024, 12, 1)
            df['Timestamp'] = [base_date + timedelta(hours=i) for i in range(len(df))]
            
        # Clean and convert data types - UPDATED COLUMN NAMES
        numeric_cols = ['Longitude', 'Latitude', 'Speed', 'Level', 'Qual', 'SNR', 'CQI', 
                       'LTERSSI', 'PSC', 'Altitude', 'DL_bitrate', 'UL_bitrate', 'PINGAVG',
                       'dist_to_node_km', 'predicted_downlink_throughput', 'predicted_uplink_throughput',
                       'anomaly_proba', 'prediction_confidence', 'risk_score']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create total throughput
        if 'DL_bitrate' in df.columns and 'UL_bitrate' in df.columns:
            df['total_throughput'] = df['DL_bitrate'] + df['UL_bitrate']
        
        # Create signal quality metrics
        if 'Level' in df.columns and 'SNR' in df.columns:
            df['signal_quality'] = (df['Level'] + df['SNR']) / 2
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_performance_score(df):
    """Calculate a composite performance score based on available metrics"""
    if df.empty:
        return df
    
    score = pd.Series(0.0, index=df.index)
    
    # Throughput component (40% weight)
    if 'DL_bitrate' in df.columns:
        dl_max = df['DL_bitrate'].max()
        if dl_max > 0:
            score += 0.3 * (df['DL_bitrate'] / dl_max)
    
    if 'UL_bitrate' in df.columns:
        ul_max = df['UL_bitrate'].max()
        if ul_max > 0:
            score += 0.1 * (df['UL_bitrate'] / ul_max)
    
    # Latency component (30% weight)
    if 'PINGAVG' in df.columns:
        lat_max = df['PINGAVG'].max()
        if lat_max > 0:
            score += 0.3 * (1 - (df['PINGAVG'] / lat_max))
    
    # Signal quality components (30% weight total)
    if 'SNR' in df.columns:
        snr_normalized = (df['SNR'] - df['SNR'].min()) / (df['SNR'].max() - df['SNR'].min())
        score += 0.15 * snr_normalized.fillna(0)
    
    if 'Level' in df.columns:
        level_normalized = (df['Level'] - df['Level'].min()) / (df['Level'].max() - df['Level'].min())
        score += 0.15 * level_normalized.fillna(0)
    
    df['performance_score'] = score * 100
    return df

def create_enhanced_performance_dashboard(df, time_col):
    """Create comprehensive performance dashboard with enhanced visualizations"""
    
    # Create a 3x3 subplot layout for better organization
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Throughput Trends Over Time', 'Performance Score Distribution',
            'Latency vs Distance to Node', 'Network Technology Performance', 'Mobility Impact Analysis',
            'Real-time Performance Radar', 'KPI Correlation Matrix', 'Anomaly Detection Timeline'
        ),
        specs=[
            [{"colspan": 2}, None, {"type": "xy"}],
            [{}, {}, {}],
            [{"type": "polar"}, {"type": "heatmap"}, {}]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # Enhanced color palette
    colors = {
        'primary': '#667eea',
        'secondary': '#764ba2', 
        'success': '#06d6a0',
        'warning': '#ffd60a',
        'danger': '#f72585',
        'info': '#4cc9f0'
    }
    
    # 1. Enhanced Throughput Trends (Row 1, Col 1-2)
    if 'DL_bitrate' in df.columns:
        # Add trend line with confidence interval
        fig.add_trace(
            go.Scatter(
                x=df[time_col], 
                y=df['DL_bitrate'],
                mode='lines+markers',
                name='Downlink Throughput',
                line=dict(color=colors['primary'], width=3),
                marker=dict(size=6),
                hovertemplate='<b>Downlink</b><br>%{y:.1f} Mbps<br>%{x}<extra></extra>'
            ), row=1, col=1
        )
        
        # Add moving average
        if len(df) > 10:
            ma_window = min(24, len(df) // 4)
            df['dl_ma'] = df['DL_bitrate'].rolling(window=ma_window, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=df[time_col], 
                    y=df['dl_ma'],
                    mode='lines',
                    name='Trend',
                    line=dict(color=colors['secondary'], width=2, dash='dash'),
                    hovertemplate='<b>Trend</b><br>%{y:.1f} Mbps<extra></extra>'
                ), row=1, col=1
            )
    
    if 'UL_bitrate' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[time_col], 
                y=df['UL_bitrate'],
                mode='lines+markers',
                name='Uplink Throughput',
                line=dict(color=colors['success'], width=3),
                marker=dict(size=6),
                hovertemplate='<b>Uplink</b><br>%{y:.1f} Mbps<br>%{x}<extra></extra>'
            ), row=1, col=1
        )
    
    # 2. Performance Score Distribution (Row 1, Col 3)
    if 'performance_score' in df.columns:
        performance_bins = pd.cut(df['performance_score'], bins=5, labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        perf_counts = performance_bins.value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                y=perf_counts.index.tolist(),
                x=perf_counts.values,
                name='Performance Distribution',
                orientation='h',
                marker_color=colors['info'],
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
            ), row=1, col=3
        )
    
    # 3. Latency vs Distance Analysis (Row 2, Col 1)
    if 'PINGAVG' in df.columns and 'dist_to_node_km' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['dist_to_node_km'], 
                y=df['PINGAVG'],
                mode='markers',
                name='Latency vs Distance',
                marker=dict(
                    size=8,
                    color=df.get('SNR', [0]*len(df)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="SNR (dB)", 
                        x=0.28, 
                        y=0.5, 
                        len=0.4,
                        thickness=15
                    )
                ),
                hovertemplate='<b>Distance:</b> %{x:.1f} km<br><b>Latency:</b> %{y:.1f} ms<extra></extra>'
            ), row=2, col=1
        )
        
        # Add trend line
        if len(df.dropna(subset=['PINGAVG', 'dist_to_node_km'])) > 5:
            z = np.polyfit(df['dist_to_node_km'].dropna(), df['PINGAVG'].dropna(), 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=df['dist_to_node_km'], 
                    y=p(df['dist_to_node_km']),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color=colors['danger'], width=2, dash='dot')
                ), row=2, col=1
            )
    
    # 4. Network Technology Performance Comparison (Row 2, Col 2)
    if 'NetworkTech' in df.columns and 'DL_bitrate' in df.columns:
        tech_performance = df.groupby('NetworkTech').agg({
            'DL_bitrate': 'mean',
            'UL_bitrate': 'mean' if 'UL_bitrate' in df.columns else lambda x: 0,
            'PINGAVG': 'mean' if 'PINGAVG' in df.columns else lambda x: 0
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=tech_performance['NetworkTech'],
                y=tech_performance['DL_bitrate'],
                name='Avg DL Throughput',
                marker_color=colors['primary'],
                hovertemplate='<b>%{x}</b><br>Avg DL: %{y:.1f} Mbps<extra></extra>'
            ), row=2, col=2
        )
    
    # 5. Mobility Impact Analysis (Row 2, Col 3)
    if 'Mobility' in df.columns and 'total_throughput' in df.columns:
        mobility_stats = df.groupby('Mobility')['total_throughput'].agg(['mean', 'std']).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=mobility_stats['Mobility'],
                y=mobility_stats['mean'],
                error_y=dict(type='data', array=mobility_stats['std']),
                name='Total Throughput by Mobility',
                marker_color=colors['info'],
                hovertemplate='<b>%{x}</b><br>Avg Total: %{y:.1f} Mbps<extra></extra>'
            ), row=2, col=3
        )
    
    # 6. Real-time Performance Radar Chart (Row 3, Col 1)
    if all(col in df.columns for col in ['DL_bitrate', 'UL_bitrate', 'PINGAVG', 'SNR', 'Level']):
        # Normalize metrics for radar chart
        metrics = ['DL Throughput', 'UL Throughput', 'Low Latency', 'SNR', 'Signal Level']
        
        # Get latest values and normalize
        latest_dl = df['DL_bitrate'].iloc[-1] / df['DL_bitrate'].max() * 100
        latest_ul = df['UL_bitrate'].iloc[-1] / df['UL_bitrate'].max() * 100
        latest_lat = (1 - df['PINGAVG'].iloc[-1] / df['PINGAVG'].max()) * 100  # Inverted for radar
        latest_snr = (df['SNR'].iloc[-1] - df['SNR'].min()) / (df['SNR'].max() - df['SNR'].min()) * 100
        latest_level = (df['Level'].iloc[-1] - df['Level'].min()) / (df['Level'].max() - df['Level'].min()) * 100
        
        values = [latest_dl, latest_ul, latest_lat, latest_snr, latest_level]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name='Current Performance',
                line=dict(color=colors['primary'], width=3),
                fillcolor=f"rgba(102, 126, 234, 0.3)"
            ), row=3, col=1
        )
    
    # 7. KPI Correlation Heatmap (Row 3, Col 2)
    numeric_cols = ['DL_bitrate', 'UL_bitrate', 'PINGAVG', 'SNR', 'Level', 'CQI']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 2:
        corr_matrix = df[available_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                showscale=True,
                hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(
                    title="Correlation", 
                    x=0.66, 
                    y=0.15, 
                    len=0.4,
                    thickness=15
                )
            ), row=3, col=2
        )
    
    # 8. Anomaly Detection Timeline (Row 3, Col 3)
    if 'anomaly_proba' in df.columns:
        # Create anomaly threshold line
        threshold = 0.5
        anomaly_mask = df['anomaly_proba'] > threshold
        
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df['anomaly_proba'],
                mode='lines+markers',
                name='Anomaly Probability',
                line=dict(color=colors['warning'], width=2),
                marker=dict(
                    size=6,
                    color=np.where(anomaly_mask, colors['danger'], colors['warning'])
                ),
                hovertemplate='<b>Anomaly Probability</b><br>%{y:.3f}<br>%{x}<extra></extra>'
            ), row=3, col=3
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=df[time_col].min(),
            y0=threshold,
            x1=df[time_col].max(),
            y1=threshold,
            line=dict(
                color=colors['danger'],
                dash="dash"
            ),
            row=3, col=3
        )

        # Add the annotation separately
        fig.add_annotation(
            text="Threshold",
            x=df[time_col].max(),
            y=threshold,
            xanchor="left",  # Positions text to the right of the anchor point
            yshift=5,        # Slight vertical offset for better visibility
            showarrow=False,
            row=3, col=3
        )
    
    # Update layout with enhanced styling
    fig.update_layout(
        height=1000,
        showlegend=True,
        template="plotly_white",
        font=dict(size=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update individual subplot layouts
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_interactive_geo_map(df):
    """Create an enhanced interactive geographical visualization"""
    if not all(col in df.columns for col in ['Latitude', 'Longitude']):
        return None
    
    # Remove NaN values
    geo_df = df.dropna(subset=['Latitude', 'Longitude'])
    if geo_df.empty:
        return None
    
    # Create the base map
    center_lat = geo_df['Latitude'].mean()
    center_lon = geo_df['Longitude'].mean()
    
    fig = go.Figure()
    
    # Add performance-based markers
    if 'performance_score' in geo_df.columns:
        fig.add_trace(go.Scattermapbox(
            lat=geo_df['Latitude'],
            lon=geo_df['Longitude'],
            mode='markers',
            marker=dict(
            size=12,
            color=geo_df['performance_score'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=dict(text="Performance Score", side="bottom"),
                thickness=15,
                x=1.0
                )
            ),
            text=geo_df.apply(lambda row: f"""
                <b>Performance Score:</b> {row.get('performance_score', 'N/A'):.1f}%<br>
                <b>DL Throughput:</b> {row.get('DL_bitrate', 'N/A'):.1f} Mbps<br>
                <b>UL Throughput:</b> {row.get('UL_bitrate', 'N/A'):.1f} Mbps<br>
                <b>Latency:</b> {row.get('PINGAVG', 'N/A'):.1f} ms<br>
                <b>Node:</b> {row.get('Node', 'Unknown')}
            """, axis=1),
            hovertemplate='%{text}<extra></extra>',
            name='Network Performance'
        ))
    
    # Add network nodes if available
    if 'Node' in geo_df.columns:
        node_locations = geo_df.groupby('Node')[['Latitude', 'Longitude']].mean().reset_index()
        
        fig.add_trace(go.Scattermapbox(
            lat=node_locations['Latitude'],
            lon=node_locations['Longitude'],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='cell-tower'
            ),
            text=node_locations['Node'],
            hovertemplate='<b>Node:</b> %{text}<extra></extra>',
            name='Network Nodes'
        ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        height=500,
        title="📍 Geographic Performance Distribution",
        showlegend=True
    )
    
    return fig

def create_advanced_predictive_charts(df, time_col):
    """Create enhanced predictive analytics visualizations"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction Accuracy Analysis', 
            'Confidence vs Performance Correlation',
            'Risk Assessment Matrix',
            'Future Performance Projection'
        ),
        specs=[[{}, {}], [{"type": "scatter"}, {}]]
    )
    
    colors = ['#667eea', '#764ba2', '#06d6a0', '#ffd60a', '#f72585']
    
    # 1. Prediction Accuracy Analysis
    if all(col in df.columns for col in ['predicted_downlink_throughput', 'DL_bitrate']):
        actual = df['DL_bitrate'].dropna()
        predicted = df['predicted_downlink_throughput'].dropna()
        
        min_len = min(len(actual), len(predicted))
        if min_len > 0:
            actual = actual.iloc[:min_len]
            predicted = predicted.iloc[:min_len]
            
            # Scatter plot for actual vs predicted
            fig.add_trace(
                go.Scatter(
                    x=actual,
                    y=predicted,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=8,
                        color=colors[0],
                        opacity=0.6
                    ),
                    hovertemplate='<b>Actual:</b> %{x:.1f} Mbps<br><b>Predicted:</b> %{y:.1f} Mbps<extra></extra>'
                ), row=1, col=1
            )
            
            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red', width=2)
                ), row=1, col=1
            )
    
    # 2. Confidence vs Performance Correlation
    if all(col in df.columns for col in ['prediction_confidence', 'performance_score']):
        fig.add_trace(
            go.Scatter(
                x=df['prediction_confidence'],
                y=df['performance_score'],
                mode='markers',
                name='Confidence vs Performance',
                marker=dict(
                    size=10,
                    color=df.get('risk_score', [0]*len(df)),
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(
    title="Risk Score",
    x=0.52,           # horizontal position relative to full figure width
    xanchor="right",  # attach the right edge at x=0.53
    y=0.84,           # vertical center (adjust this)
    len=0.4           # vertical size of colorbar (adjust this)
)
                ),
                hovertemplate='<b>Confidence:</b> %{x:.3f}<br><b>Performance:</b> %{y:.1f}%<extra></extra>'
            ), row=1, col=2
        )
    
    # 3. Risk Assessment Scatter Matrix
    if all(col in df.columns for col in ['risk_score', 'anomaly_proba']):
        # Create risk categories
        df['risk_category'] = pd.cut(
            df['risk_score'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        for category in df['risk_category'].cat.categories:
            mask = df['risk_category'] == category
            color_map = {'Low': colors[2], 'Medium': colors[3], 'High': colors[4]}
            
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, 'anomaly_proba'],
                    y=df.loc[mask, 'risk_score'],
                    mode='markers',
                    name=f'{category} Risk',
                    marker=dict(
                        size=8,
                        color=color_map[category],
                        opacity=0.7
                    )
                ), row=2, col=1
            )
    
    # 4. Future Performance Projection
    if 'predicted_downlink_throughput' in df.columns:
        # Create time-based projection
        future_times = pd.date_range(
            start=df[time_col].max(), 
            periods=24, 
            freq='H'
        )[1:]  # Exclude the first point to avoid duplication
        
        # Simple trend projection (you could enhance this with actual ML models)
        recent_data = df.tail(48)  # Last 48 hours
        if len(recent_data) > 1:
            trend = np.polyfit(range(len(recent_data)), recent_data['predicted_downlink_throughput'], 1)
            future_values = np.polyval(trend, range(len(recent_data), len(recent_data) + 24))
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=df[time_col].tail(48),
                    y=df['predicted_downlink_throughput'].tail(48),
                    mode='lines+markers',
                    name='Historical Predictions',
                    line=dict(color=colors[0], width=3)
                ), row=2, col=2
            )
            
            # Future projection
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=future_values,
                    mode='lines+markers',
                    name='Future Projection',
                    line=dict(color=colors[1], width=3, dash='dash'),
                    marker=dict(symbol='diamond')
                ), row=2, col=2
            )
    
    fig.update_layout(
        height=700,
        title="🔮 Advanced Predictive Analytics Dashboard",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

# Main app
def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">📡 Advanced Mobile Network Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data available. Please check if 'dashboard_ready_with_predictions.csv' exists and is properly formatted.")
        return
    
    # Calculate performance scores
    df = calculate_performance_score(df)
    
    # Determine time column
    time_col = 'Timestamp'
    
    # Sidebar filters (keeping original filters)
    st.sidebar.title('🔧 Filter Controls')
    st.sidebar.markdown("---")
    
    # Advanced filters
    with st.sidebar.expander("📊 Data Filters", expanded=True):
        # Operator filter
        if 'Operatorname' in df.columns:
            operators = df['Operatorname'].unique()
            selected_operator = st.multiselect('📱 Operator', options=operators, default=operators)
        else:
            selected_operator = []
        
        # Network technology filter
        if 'NetworkTech' in df.columns:
            network_techs = df['NetworkTech'].unique()
            selected_tech = st.multiselect('🔗 Network Technology', options=network_techs, default=network_techs)
        else:
            selected_tech = []
        
        # Mobility filter
        if 'Mobility' in df.columns:
            mobilities = df['Mobility'].unique()
            selected_mobility = st.multiselect('🚶 Mobility Type', options=mobilities, default=mobilities)
        else:
            selected_mobility = []
        
        # Date range filter
        if time_col in df.columns and not df[time_col].isna().all():
            min_date = df[time_col].min().date()
            max_date = df[time_col].max().date()
            date_range = st.date_input('📅 Date Range', [min_date, max_date])
        else:
            date_range = []
        
        # Session filter
        if 'SessionID' in df.columns:
            session_ids = sorted(df['SessionID'].unique())
            selected_sessions = st.multiselect('📄 Session IDs', 
                                             options=session_ids, 
                                             default=session_ids[:10] if len(session_ids) > 10 else session_ids)
        else:
            selected_sessions = []
    
    # Performance thresholds
    with st.sidebar.expander("⚙️ Performance Thresholds", expanded=False):
        latency_threshold = st.slider('Max Latency (ms)', 0, 500, 200)
        min_throughput = st.slider('Min Throughput (Mbps)', 0, 200, 50)
        min_snr = st.slider('Min SNR (dB)', -20, 30, 0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_operator and 'Operatorname' in df.columns:
        filtered_df = filtered_df[filtered_df['Operatorname'].isin(selected_operator)]
    if selected_tech and 'NetworkTech' in df.columns:
        filtered_df = filtered_df[filtered_df['NetworkTech'].isin(selected_tech)]
    if selected_mobility and 'Mobility' in df.columns:
        filtered_df = filtered_df[filtered_df['Mobility'].isin(selected_mobility)]
    if selected_sessions and 'SessionID' in df.columns:
        filtered_df = filtered_df[filtered_df['SessionID'].isin(selected_sessions)]
    
    # Date filter
    if date_range and len(date_range) == 2 and time_col in filtered_df.columns:
        try:
            filtered_df = filtered_df[
                (filtered_df[time_col].dt.date >= date_range[0]) &
                (filtered_df[time_col].dt.date <= date_range[1])
            ]
        except Exception as e:
            st.warning(f"Date filtering failed: {str(e)}. Showing all data.")
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Dashboard overview
    st.markdown("### 📈 Performance Overview")
    
    # KPI Metrics with better styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_dl = filtered_df.get('DL_bitrate', pd.Series([0])).mean()
        delta_dl = avg_dl - df.get('DL_bitrate', pd.Series([0])).mean()
        st.metric('🔥 Avg Downlink', f"{avg_dl:.1f} Mbps", f"{delta_dl:+.1f}")
    
    with col2:
        avg_ul = filtered_df.get('UL_bitrate', pd.Series([0])).mean()
        delta_ul = avg_ul - df.get('UL_bitrate', pd.Series([0])).mean()
        st.metric('📤 Avg Uplink', f"{avg_ul:.1f} Mbps", f"{delta_ul:+.1f}")
    
    with col3:
        avg_latency = filtered_df.get('PINGAVG', pd.Series([0])).mean()
        delta_latency = avg_latency - df.get('PINGAVG', pd.Series([0])).mean()
        st.metric('⏱️ Avg Latency', f"{avg_latency:.1f} ms", f"{delta_latency:+.1f}")
    
    with col4:
        avg_perf = filtered_df.get('performance_score', pd.Series([0])).mean()
        delta_perf = avg_perf - df.get('performance_score', pd.Series([0])).mean()
        st.metric('🎯 Performance Score', f"{avg_perf:.1f}%", f"{delta_perf:+.1f}")
    
    with col5:
        # Calculate anomalies based on thresholds
        anomaly_conditions = []
        if 'PINGAVG' in filtered_df.columns:
            anomaly_conditions.append(filtered_df['PINGAVG'] > latency_threshold)
        if 'DL_bitrate' in filtered_df.columns:
            anomaly_conditions.append(filtered_df['DL_bitrate'] < min_throughput)
        if 'SNR' in filtered_df.columns:
            anomaly_conditions.append(filtered_df['SNR'] < min_snr)
        if 'Level' in filtered_df.columns:
            anomaly_conditions.append(filtered_df['Level'] < -100)
        
        if anomaly_conditions:
            anomaly_count = len(filtered_df[np.logical_or.reduce(anomaly_conditions)])
        else:
            anomaly_count = 0
        
        anomaly_rate = (anomaly_count/len(filtered_df)*100) if len(filtered_df) > 0 else 0
        st.metric('🚨 Rule-Based Anomalies', anomaly_count, f"{anomaly_rate:.1f}%")
    
    # Enhanced main dashboard
    st.markdown("### 📊 Comprehensive Performance Analytics")
    
    if len(filtered_df) > 0:
        enhanced_fig = create_enhanced_performance_dashboard(filtered_df, time_col)
        st.plotly_chart(enhanced_fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 
                                   'drawcircle', 'drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'comprehensive_network_analytics',
                'height': 1000,
                'width': 1400,
                'scale': 2
            }
        })
    
    # Geographic Visualization
    st.markdown("### 🗺️ Geographic Performance Analysis")
    geo_map = create_interactive_geo_map(filtered_df)
    if geo_map:
        st.plotly_chart(geo_map, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'geographic_performance',
                'height': 500,
                'width': 1200,
                'scale': 2
            }
        })
    else:
        st.info("Geographic data not available for mapping.")
    
    # Predictive Analytics Section
    st.markdown("### 🔮 Advanced Predictive Analytics & AI Insights")
    
    # Enhanced Predictive KPI metrics
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

    with pred_col1:
        if 'predicted_downlink_throughput' in filtered_df.columns:
            pred_dl_avg = filtered_df['predicted_downlink_throughput'].mean()
            actual_dl_avg = filtered_df.get('DL_bitrate', pd.Series([0])).mean()
            accuracy = 100 - abs(pred_dl_avg - actual_dl_avg) / actual_dl_avg * 100 if actual_dl_avg > 0 else 0
            st.metric('🎯 Predicted DL Throughput', f"{pred_dl_avg:.1f} Mbps", f"Accuracy: {accuracy:.1f}%")
        else:
            st.metric('🎯 Predicted DL Throughput', "N/A", "Model not available")

    with pred_col2:
        if 'predicted_uplink_throughput' in filtered_df.columns:
            pred_ul_avg = filtered_df['predicted_uplink_throughput'].mean()
            actual_ul_avg = filtered_df.get('UL_bitrate', pd.Series([0])).mean()
            accuracy = 100 - abs(pred_ul_avg - actual_ul_avg) / actual_ul_avg * 100 if actual_ul_avg > 0 else 0
            st.metric('🎯 Predicted UL Throughput', f"{pred_ul_avg:.1f} Mbps", f"Accuracy: {accuracy:.1f}%")
        else:
            st.metric('🎯 Predicted UL Throughput', "N/A", "Model not available")
    
    with pred_col3:
        if 'prediction_confidence' in filtered_df.columns:
            avg_confidence = filtered_df['prediction_confidence'].mean() * 100
            confidence_level = "High" if avg_confidence > 80 else "Medium" if avg_confidence > 60 else "Low"
            st.metric('🎲 Model Confidence', f"{avg_confidence:.1f}%", confidence_level)
    
    with pred_col4:
        if 'risk_score' in filtered_df.columns:
            avg_risk = filtered_df['risk_score'].mean() * 100
            risk_level = "Low" if avg_risk < 30 else "Medium" if avg_risk < 70 else "High"
            st.metric('⚠️ Risk Score', f"{avg_risk:.1f}%", risk_level)
    
    # Advanced Predictive Charts
    pred_charts = create_advanced_predictive_charts(filtered_df, time_col)
    st.plotly_chart(pred_charts, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'predictive_analytics',
            'height': 700,
            'width': 1400,
            'scale': 2
        }
    })
    
    # Enhanced predictive analysis tabs
    pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs(["📈 Forecast Trends", "🎯 Model Performance", "⚡ Anomaly Prediction", "💡 AI Recommendations"])
    
    with pred_tab1:
        st.markdown("#### Enhanced Predicted vs Actual KPI Trends")

        if 'predicted_downlink_throughput' in filtered_df.columns and 'DL_bitrate' in filtered_df.columns:
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=('Downlink Throughput Forecast', 'Prediction Error Analysis',
                                            'Uplink Throughput Forecast', 'Confidence Intervals'),
                              specs=[[{}, {}], [{}, {}]])

            # Enhanced Downlink comparison with confidence bands
            fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['DL_bitrate'],
                                   mode='lines+markers', name='Actual DL', 
                                   line=dict(color='#1f77b4', width=3),
                                   marker=dict(size=6)), row=1, col=1)
            fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['predicted_downlink_throughput'],
                                   mode='lines+markers', name='Predicted DL', 
                                   line=dict(color='#ff7f0e', dash='dash', width=3),
                                   marker=dict(size=6, symbol='diamond')), row=1, col=1)

            # Add confidence intervals if available
            if 'prediction_confidence' in filtered_df.columns:
                confidence = filtered_df['prediction_confidence']
                upper_bound = filtered_df['predicted_downlink_throughput'] * (1 + (1 - confidence))
                lower_bound = filtered_df['predicted_downlink_throughput'] * (1 - (1 - confidence))
                
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=upper_bound,
                                       mode='lines', name='Upper Bound', 
                                       line=dict(width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=lower_bound,
                                       mode='lines', name='Lower Bound', 
                                       line=dict(width=0), 
                                       fill='tonexty', fillcolor='rgba(255,127,14,0.2)',
                                       showlegend=False), row=1, col=1)

            # Prediction Error Analysis
            error = filtered_df['DL_bitrate'] - filtered_df['predicted_downlink_throughput']
            fig.add_trace(go.Scatter(x=filtered_df[time_col], y=error,
                                   mode='lines+markers', name='Prediction Error',
                                   line=dict(color='red', width=2),
                                   marker=dict(size=4)), row=1, col=2)
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

            # Uplink comparison if available
            if 'predicted_uplink_throughput' in filtered_df.columns and 'UL_bitrate' in filtered_df.columns:
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['UL_bitrate'],
                                       mode='lines+markers', name='Actual UL', 
                                       line=dict(color='#2ca02c', width=3)), row=2, col=1)
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['predicted_uplink_throughput'],
                                       mode='lines+markers', name='Predicted UL', 
                                       line=dict(color='#d62728', dash='dash', width=3)), row=2, col=1)

            # Confidence over time
            if 'prediction_confidence' in filtered_df.columns:
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['prediction_confidence'],
                                       mode='lines+markers', name='Model Confidence',
                                       line=dict(color='purple', width=2),
                                       marker=dict(size=5)), row=2, col=2)

            fig.update_layout(height=700, title_text="🔮 Enhanced AI-Powered KPI Forecasting")
            st.plotly_chart(fig, use_container_width=True)
    
    with pred_tab2:
        st.markdown("#### Enhanced Model Performance & Accuracy Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced confidence distribution with statistical info
            if 'prediction_confidence' in filtered_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=filtered_df['prediction_confidence'], 
                                         nbinsx=30, name="Confidence Distribution",
                                         marker_color='rgba(102, 126, 234, 0.7)',
                                         opacity=0.8))
                
                # Add mean line
                mean_conf = filtered_df['prediction_confidence'].mean()
                fig.add_vline(x=mean_conf, line_dash="dash", line_color="red",
                            annotation_text=f"Mean: {mean_conf:.3f}")
                
                fig.update_layout(title="🎯 Prediction Confidence Distribution",
                                xaxis_title="Confidence Score",
                                yaxis_title="Frequency",
                                showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model performance by node with enhanced visualization
            if 'Node' in filtered_df.columns and 'prediction_confidence' in filtered_df.columns:
                node_stats = filtered_df.groupby('Node').agg({
                    'prediction_confidence': ['mean', 'std', 'count'],
                    'risk_score': 'mean' if 'risk_score' in filtered_df.columns else lambda x: 0
                }).reset_index()

                node_stats.columns = ['Node', 'Avg_Confidence', 'Std_Confidence', 'Count', 'Avg_Risk']
                node_stats = node_stats.head(15)  # Show top 15 nodes

                fig = go.Figure()
                fig.add_trace(go.Bar(x=node_stats['Node'], y=node_stats['Avg_Confidence'],
                                   error_y=dict(type='data', array=node_stats['Std_Confidence']),
                                   marker=dict(
                                       color=node_stats['Avg_Risk'],
                                       colorscale='RdYlGn_r',
                                       colorbar=dict(title='Avg Risk')  # Optional: adds a colorbar for the risk-based coloring
                                   ),
                                   name='Confidence by Node'))

                fig.update_layout(title="📊 Model Confidence by Node",
                                xaxis_title="Node",
                                yaxis_title="Average Confidence",
                                xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced model accuracy metrics with more statistics
        if 'predicted_downlink_throughput' in filtered_df.columns and 'DL_bitrate' in filtered_df.columns:
            st.markdown("#### 📊 Detailed Model Accuracy Statistics")
            
            actual = filtered_df['DL_bitrate'].dropna()
            predicted = filtered_df['predicted_downlink_throughput'].dropna()
            
            if len(actual) > 0 and len(predicted) > 0:
                min_len = min(len(actual), len(predicted))
                actual = actual.iloc[:min_len]
                predicted = predicted.iloc[:min_len]
                
                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                r2 = stats.pearsonr(actual, predicted)[0]**2
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Mean Absolute Error", f"{mae:.2f} Mbps")
                with metric_col2:
                    st.metric("Root Mean Square Error", f"{rmse:.2f} Mbps")
                with metric_col3:
                    st.metric("Mean Absolute Percentage Error", f"{mape:.1f}%")
                with metric_col4:
                    st.metric("R-squared Score", f"{r2:.3f}")
                
                # Enhanced residual plot
                residuals = actual - predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=predicted, y=residuals,
                                       mode='markers', name='Residuals',
                                       marker=dict(color='blue', opacity=0.6)))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(title="📈 Residual Analysis",
                                xaxis_title="Predicted Values",
                                yaxis_title="Residuals")
                st.plotly_chart(fig, use_container_width=True)
    
    with pred_tab3:
        st.markdown("#### 🚨 Enhanced AI-Powered Anomaly Detection & Risk Assessment")
        
        if 'anomaly_proba' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced high-risk nodes analysis
                high_risk_threshold = filtered_df['anomaly_proba'].quantile(0.8)
                high_risk_nodes = filtered_df[filtered_df['anomaly_proba'] > high_risk_threshold]
                
                st.markdown(f"**🔥 High-Risk Nodes Analysis ({len(high_risk_nodes)} detected)**")
                if not high_risk_nodes.empty:
                    risk_summary = high_risk_nodes.groupby('Node').agg({
                        'anomaly_proba': ['mean', 'max', 'count'],
                        'risk_score': 'mean' if 'risk_score' in filtered_df.columns else lambda x: 0
                    }).reset_index()
                    risk_summary.columns = ['Node', 'Avg_Anomaly_Prob', 'Max_Anomaly_Prob', 'Anomaly_Count', 'Risk_Score']
                    risk_summary = risk_summary.sort_values('Avg_Anomaly_Prob', ascending=False)
                    
                    # Enhanced risk visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=risk_summary['Avg_Anomaly_Prob'], 
                                           y=risk_summary['Risk_Score'],
                                           mode='markers+text',
                                           text=risk_summary['Node'],
                                           textposition="top center",
                                           marker=dict(size=risk_summary['Anomaly_Count']*2,
                                                     color=risk_summary['Max_Anomaly_Prob'],
                                                     colorscale='Reds',
                                                     showscale=True)))
                    fig.update_layout(title="🎯 Risk vs Anomaly Probability Matrix",
                                    xaxis_title="Average Anomaly Probability",
                                    yaxis_title="Risk Score")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Enhanced anomaly probability distribution with statistics
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=filtered_df['anomaly_proba'], nbinsx=50,
                                         name="Anomaly Probability",
                                         marker_color='rgba(255, 99, 71, 0.7)'))
                fig.add_vline(x=high_risk_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"High Risk Threshold: {high_risk_threshold:.3f}")
                
                mean_anomaly = filtered_df['anomaly_proba'].mean()
                fig.add_vline(x=mean_anomaly, line_dash="dot", line_color="blue",
                            annotation_text=f"Mean: {mean_anomaly:.3f}")
                
                fig.update_layout(title="📊 Enhanced Anomaly Probability Distribution",
                                xaxis_title="Anomaly Probability",
                                yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced proactive alerts with more detail
        if 'risk_score' in filtered_df.columns:
            st.markdown("#### 🚨 Intelligent Proactive Network Alerts")
            
            critical_nodes = filtered_df[filtered_df['risk_score'] > 0.7]
            warning_nodes = filtered_df[(filtered_df['risk_score'] > 0.4) & (filtered_df['risk_score'] <= 0.7)]
            
            if not critical_nodes.empty:
                critical_details = critical_nodes.groupby('Node')['risk_score'].max().sort_values(ascending=False).head(5)
                alert_text = f"🔴 CRITICAL: {len(critical_nodes)} measurements from {len(critical_details)} nodes require immediate attention"
                st.markdown(f"<div class='alert-box danger-alert'><b>{alert_text}</b><br>Top nodes: {', '.join([f'{node} ({score:.2f})' for node, score in critical_details.items()])}</div>", 
                           unsafe_allow_html=True)
                
            if not warning_nodes.empty:
                warning_details = warning_nodes.groupby('Node')['risk_score'].max().sort_values(ascending=False).head(5)
                alert_text = f"🟡 WARNING: {len(warning_nodes)} measurements from {len(warning_details)} nodes show degrading performance"
                st.markdown(f"<div class='alert-box warning-alert'><b>{alert_text}</b><br>Watch nodes: {', '.join([f'{node} ({score:.2f})' for node, score in warning_details.items()])}</div>", 
                           unsafe_allow_html=True)
            
            if critical_nodes.empty and warning_nodes.empty:
                st.markdown("<div class='alert-box success-alert'><b>✅ All nodes operating within normal parameters - Network health is optimal</b></div>", 
                           unsafe_allow_html=True)
    
    with pred_tab4:
        st.markdown("#### 🤖 Enhanced AI-Driven Network Optimization Recommendations")
        
        # Generate more sophisticated recommendations
        recommendations = []
        
        if 'predicted_downlink_throughput' in filtered_df.columns:
            low_throughput_nodes = filtered_df[filtered_df['predicted_downlink_throughput'] < filtered_df['predicted_downlink_throughput'].quantile(0.2)]
            if not low_throughput_nodes.empty:
                avg_predicted = low_throughput_nodes['predicted_downlink_throughput'].mean()
                recommendations.append({
                    'priority': 'High',
                    'category': '📡 Throughput Optimization',
                    'description': f'Optimize {len(low_throughput_nodes.Node.unique())} nodes with predicted low throughput (avg: {avg_predicted:.1f} Mbps)',
                    'action': 'Review antenna configuration, check for interference, consider carrier aggregation',
                    'impact': 'Expected 15-25% throughput improvement',
                    'timeline': '2-4 weeks'
                })
        
        if 'risk_score' in filtered_df.columns:
            high_risk_nodes = filtered_df[filtered_df['risk_score'] > 0.6]
            if not high_risk_nodes.empty:
                avg_risk = high_risk_nodes['risk_score'].mean()
                recommendations.append({
                    'priority': 'Critical',
                    'category': '🚨 Risk Mitigation',
                    'description': f'Immediate intervention required for {len(high_risk_nodes.Node.unique())} high-risk nodes (avg risk: {avg_risk:.2f})',
                    'action': 'Schedule maintenance, backup traffic routing, monitor closely',
                    'impact': 'Prevent potential service outages',
                    'timeline': 'Immediate'
                })
        
        if 'anomaly_proba' in filtered_df.columns:
            anomaly_nodes = filtered_df[filtered_df['anomaly_proba'] > 0.5]
            if not anomaly_nodes.empty:
                avg_anomaly = anomaly_nodes['anomaly_proba'].mean()
                recommendations.append({
                    'priority': 'Medium',
                    'category': '⚡ Anomaly Prevention',
                    'description': f'Proactive maintenance for {len(anomaly_nodes.Node.unique())} nodes showing anomaly patterns (avg prob: {avg_anomaly:.2f})',
                    'action': 'Predictive maintenance scheduling, parameter optimization',
                    'impact': 'Reduce future anomalies by 40-60%',
                    'timeline': '1-3 weeks'
                })
        
        # Enhanced network technology optimization
        if 'NetworkTech' in filtered_df.columns and 'predicted_downlink_throughput' in filtered_df.columns:
            tech_performance = filtered_df.groupby('NetworkTech')['predicted_downlink_throughput'].mean()
            underperforming_tech = tech_performance[tech_performance < tech_performance.median()].index
            if len(underperforming_tech) > 0:
                recommendations.append({
                    'priority': 'Medium',
                    'category': '🔧 Technology Upgrade',
                    'description': f'Consider upgrading {", ".join(underperforming_tech)} technology (performance below median)',
                    'action': 'Evaluate 5G deployment, optimize existing technology parameters',
                    'impact': 'Expected 30-50% performance boost',
                    'timeline': '3-6 months'
                })
        
        # Display enhanced recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}
                color = priority_colors.get(rec['priority'], '#6c757d')
                
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 1.5rem; margin: 1rem 0; 
           background: linear-gradient(135deg, rgba(30,30,30,0.95), rgba(50,50,50,0.9)); 
           color: white;
           border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="color: {color}; margin: 0;">{rec['category']} - {rec['priority']} Priority</h4>
                        <span style="background: {color}; color: white; padding: 0.25rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">
                            {rec.get('timeline', 'TBD')}
                        </span>
                    </div>
                    <p style="margin: 0.5rem 0; font-weight: 500;"><b>Issue:</b> {rec['description']}</p>
                    <p style="margin: 0.5rem 0;"><b>Recommended Action:</b> {rec['action']}</p>
                    <p style="margin: 0; color: #28a745; font-weight: 500;"><b>Expected Impact:</b> {rec.get('impact', 'Performance improvement expected')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("✅ **No immediate optimization recommendations at this time. Network performance is within expected parameters.**")
        
        # Enhanced future predictions with confidence metrics
        if 'predicted_downlink_throughput' in filtered_df.columns:
            st.markdown("#### 📅 Advanced Short-term Network Performance Forecast")
            
            recent_data = filtered_df.tail(24)
            if len(recent_data) > 1:
                # Calculate multiple trend metrics
                dl_trend = (recent_data['predicted_downlink_throughput'].iloc[-1] - recent_data['predicted_downlink_throughput'].iloc[0]) / len(recent_data)
                dl_volatility = recent_data['predicted_downlink_throughput'].std()
                trend_direction = "📈 Improving" if dl_trend > 0.5 else "📉 Declining" if dl_trend < -0.5 else "➡️ Stable"
                
                # Calculate forecast confidence
                if 'prediction_confidence' in recent_data.columns:
                    forecast_confidence = recent_data['prediction_confidence'].mean()
                    confidence_text = f"High ({forecast_confidence:.1%})" if forecast_confidence > 0.8 else f"Medium ({forecast_confidence:.1%})" if forecast_confidence > 0.6 else f"Low ({forecast_confidence:.1%})"
                
                volatility_level = "High" if dl_volatility > 10 else "Medium" if dl_volatility > 5 else "Low"
                
                st.info(f"""
                **Network Performance Trend:** {trend_direction}
                
                **Expected Changes in Next 24 Hours:**
                - Downlink throughput trend: {dl_trend:+.2f} Mbps per hour
                - Volatility: {volatility_level} (std: {dl_volatility:.2f} Mbps)
                - Forecast Confidence: {confidence_text}
                - Recommended monitoring frequency: {'High (every 15 min)' if abs(dl_trend) > 1 or volatility_level == 'High' else 'Normal (hourly)'}
                """)
    
    # Additional analysis (improved from old version with enhanced plots)
    st.markdown("### 📊 Advanced Analytics")
    
    col1, col2 = st.columns(2)

    with col1:
        if 'NetworkTech' in filtered_df.columns and 'DL_bitrate' in filtered_df.columns:
            fig = px.box(filtered_df, x='NetworkTech', y='DL_bitrate', 
                        title="Downlink Throughput by Network Technology",
                        labels={'DL_bitrate': 'Throughput (Mbps)'},
                        color='NetworkTech',  # Added color for better distinction
                        notched=True)  # Added notches for better comparison
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'throughput_by_network_tech',
                    'scale': 2
                }
            })

    with col2:
        if 'Mobility' in filtered_df.columns and 'PINGAVG' in filtered_df.columns:
            fig = px.violin(filtered_df, x='Mobility', y='PINGAVG', 
                           title="Latency Distribution by Mobility Type",
                           labels={'PINGAVG': 'Latency (ms)'},
                           color='Mobility',  # Added color
                           box=True,  # Show box inside violin for summary stats
                           points='outliers')  # Show outliers
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'latency_by_mobility',
                    'scale': 2
                }
            })

if __name__ == "__main__":
    main()
