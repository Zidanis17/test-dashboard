import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mobile Network Analytics Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .warning-alert {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    .danger-alert {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
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
                       'LTERSSI', 'PSC', 'Altitude', 'DL_bitrate', 'UL_bitrate', 'PINGAVG',  # Changed from y_downlink_throughput, y_uplink_throughput, y_latency
                       'dist_to_node_km', 'predicted_downlink_throughput', 'predicted_uplink_throughput',
                       'anomaly_proba', 'prediction_confidence', 'risk_score']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create total throughput - UPDATED COLUMN NAMES
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
    
    # Throughput component (40% weight) - UPDATED COLUMN NAMES
    if 'DL_bitrate' in df.columns:  # Changed from y_downlink_throughput
        dl_max = df['DL_bitrate'].max()
        if dl_max > 0:
            score += 0.3 * (df['DL_bitrate'] / dl_max)
    
    if 'UL_bitrate' in df.columns:  # Changed from y_uplink_throughput
        ul_max = df['UL_bitrate'].max()
        if ul_max > 0:
            score += 0.1 * (df['UL_bitrate'] / ul_max)
    
    # Latency component (30% weight) - UPDATED COLUMN NAME
    if 'PINGAVG' in df.columns:  # Changed from y_latency
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

def create_performance_trends_chart(df, time_col):
    """Create comprehensive performance trends visualization"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Downlink Throughput (Mbps)', 'Uplink Throughput (Mbps)', 
                       'Latency (ms)', 'Signal Quality (SNR)',
                       'Signal Level (dBm)', 'Network Technology Distribution'),
        specs=[[{}, {}], [{}, {}], [{}, {"type": "pie"}]]
    )
    
    # Throughput trends - UPDATED COLUMN NAMES
    if 'DL_bitrate' in df.columns:  # Changed from y_downlink_throughput
        fig.add_trace(go.Scatter(x=df[time_col], y=df['DL_bitrate'], 
                                mode='lines+markers', name='Downlink Throughput', 
                                line=dict(color='#1f77b4')), row=1, col=1)
    
    if 'UL_bitrate' in df.columns:  # Changed from y_uplink_throughput
        fig.add_trace(go.Scatter(x=df[time_col], y=df['UL_bitrate'], 
                                mode='lines+markers', name='Uplink Throughput', 
                                line=dict(color='#2ca02c')), row=1, col=2)
    
    # Latency - UPDATED COLUMN NAME
    if 'PINGAVG' in df.columns:  # Changed from y_latency
        fig.add_trace(go.Scatter(x=df[time_col], y=df['PINGAVG'], 
                                mode='lines+markers', name='Latency', 
                                line=dict(color='#ff7f0e')), row=2, col=1)
    
    # SNR
    if 'SNR' in df.columns:
        fig.add_trace(go.Scatter(x=df[time_col], y=df['SNR'], 
                                mode='lines+markers', name='SNR', 
                                line=dict(color='#9467bd')), row=2, col=2)
    
    # Signal Level
    if 'Level' in df.columns:
        fig.add_trace(go.Scatter(x=df[time_col], y=df['Level'], 
                                mode='lines+markers', name='Signal Level', 
                                line=dict(color='#8c564b')), row=3, col=1)
    
    # Network Technology Distribution
    if 'NetworkTech' in df.columns:
        tech_counts = df['NetworkTech'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        fig.add_trace(go.Pie(labels=tech_counts.index, values=tech_counts.values, 
                            name="Network Tech", 
                            marker=dict(colors=colors[:len(tech_counts)])), row=3, col=2)
    
    fig.update_layout(height=900, title_text="Network Performance Metrics Over Time", showlegend=False)
    return fig

def create_anomaly_heatmap(df):
    """Create anomaly detection heatmap"""
    if 'anomaly_flag' not in df.columns:
        return None
    
    # Create pivot table for heatmap
    if 'Node' in df.columns and 'hour' in df.columns:
        heatmap_data = df.pivot_table(values='anomaly_flag', index='Node', columns='hour', aggfunc='mean', fill_value=0)
        
        fig = px.imshow(heatmap_data, 
                       title="Anomaly Detection Heatmap (by Node and Hour)",
                       labels=dict(x="Hour", y="Node", color="Anomaly Rate"),
                       color_continuous_scale="Reds")
        return fig
    return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📡 Mobile Network Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data available. Please check if 'dashboard_ready_with_predictions.csv' exists and is properly formatted.")
        return
    
    # Calculate performance scores
    df = calculate_performance_score(df)
    
    # Determine time column
    time_col = 'Timestamp'
    
    # Sidebar filters
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
    
    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_dl = filtered_df.get('DL_bitrate', pd.Series([0])).mean()  # Changed from y_downlink_throughput
        st.metric('📥 Avg Downlink', f"{avg_dl:.1f} Mbps", f"{avg_dl - df.get('DL_bitrate', pd.Series([0])).mean():.1f}")
    
    with col2:
        avg_ul = filtered_df.get('UL_bitrate', pd.Series([0])).mean()  # Changed from y_uplink_throughput
        st.metric('📤 Avg Uplink', f"{avg_ul:.1f} Mbps", f"{avg_ul - df.get('UL_bitrate', pd.Series([0])).mean():.1f}")
    
    with col3:
        avg_latency = filtered_df.get('PINGAVG', pd.Series([0])).mean()  # Changed from y_latency
        st.metric('⏱️ Avg Latency', f"{avg_latency:.1f} ms", f"{avg_latency - df.get('PINGAVG', pd.Series([0])).mean():.1f}")
    
    with col4:
        avg_perf = filtered_df.get('performance_score', pd.Series([0])).mean()
        st.metric('🎯 Performance Score', f"{avg_perf:.1f}%", f"{avg_perf - df.get('performance_score', pd.Series([0])).mean():.1f}")
    
    with col5:
        # Calculate anomalies based on thresholds
        anomaly_conditions = []
        if 'PINGAVG' in filtered_df.columns:  # Changed from y_latency
            anomaly_conditions.append(filtered_df['PINGAVG'] > latency_threshold)
        if 'DL_bitrate' in filtered_df.columns:  # Changed from y_downlink_throughput
            anomaly_conditions.append(filtered_df['DL_bitrate'] < min_throughput)
        if 'SNR' in filtered_df.columns:
            anomaly_conditions.append(filtered_df['SNR'] < min_snr)
        if 'Level' in filtered_df.columns:
            anomaly_conditions.append(filtered_df['Level'] < -100)  # Poor signal level
        
        if anomaly_conditions:
            anomaly_count = len(filtered_df[np.logical_or.reduce(anomaly_conditions)])
        else:
            anomaly_count = 0
        
        st.metric('🚨 Rule-Based Anomalies', anomaly_count, f"{(anomaly_count/len(filtered_df)*100):.1f}%")
    
    # Performance trends
    st.markdown("### 📊 Detailed Performance Analysis")
    
    if len(filtered_df) > 0:
        trends_fig = create_performance_trends_chart(filtered_df, time_col)
        st.plotly_chart(trends_fig, use_container_width=True)
    
    # =================
    # PREDICTIVE ANALYTICS SECTION - ADD THIS HERE
    # =================
    
    # Predictive Analytics Section
    st.markdown("### 🔮 Predictive Analytics & AI-Driven Insights")
    
    # Predictive KPI metrics
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
            st.metric('🎲 Model Confidence', f"{avg_confidence:.1f}%", 
                     "High" if avg_confidence > 80 else "Medium" if avg_confidence > 60 else "Low")
    
    with pred_col4:
        if 'risk_score' in filtered_df.columns:
            avg_risk = filtered_df['risk_score'].mean() * 100
            risk_level = "Low" if avg_risk < 30 else "Medium" if avg_risk < 70 else "High"
            st.metric('⚠️ Risk Score', f"{avg_risk:.1f}%", risk_level)
    
    # Predictive analysis tabs
    pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs(["📈 Forecast Trends", "🎯 Model Performance", "⚡ Anomaly Prediction", "💡 AI Recommendations"])
    
    with pred_tab1:
        st.markdown("#### Predicted vs Actual KPI Trends")

        if 'predicted_downlink_throughput' in filtered_df.columns and 'DL_bitrate' in filtered_df.columns:  # Changed from y_downlink_throughput
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Downlink Throughput Forecast', 'Uplink Throughput Forecast'))

            # Downlink comparison
            fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['DL_bitrate'],  # Changed from y_downlink_throughput
                                   mode='lines+markers', name='Actual DL', line=dict(color='#1f77b4')), row=1, col=1)
            fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['predicted_downlink_throughput'],
                                   mode='lines+markers', name='Predicted DL', line=dict(color='#ff7f0e', dash='dash')), row=1, col=1)

            # Uplink comparison if available
            if 'predicted_uplink_throughput' in filtered_df.columns and 'UL_bitrate' in filtered_df.columns:  # Changed from y_uplink_throughput
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['UL_bitrate'],  # Changed from y_uplink_throughput
                                       mode='lines+markers', name='Actual UL', line=dict(color='#2ca02c')), row=2, col=1)
                fig.add_trace(go.Scatter(x=filtered_df[time_col], y=filtered_df['predicted_uplink_throughput'],
                                       mode='lines+markers', name='Predicted UL', line=dict(color='#d62728', dash='dash')), row=2, col=1)

            fig.update_layout(height=600, title_text="AI-Powered KPI Forecasting")
            st.plotly_chart(fig, use_container_width=True)
    
    with pred_tab2:
        st.markdown("#### Model Performance & Accuracy Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'prediction_confidence' in filtered_df.columns:
                fig = px.histogram(filtered_df, x='prediction_confidence', 
                                 title="Prediction Confidence Distribution",
                                 labels={'prediction_confidence': 'Confidence Score'})
                fig.update_traces(marker_color='#9467bd')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Node' in filtered_df.columns and 'prediction_confidence' in filtered_df.columns:
                node_confidence = filtered_df.groupby('Node')['prediction_confidence'].mean().reset_index()
                fig = px.bar(node_confidence.head(10), x='Node', y='prediction_confidence',
                           title="Model Confidence by Node",
                           labels={'prediction_confidence': 'Avg Confidence'})
                fig.update_traces(marker_color='#17becf')
                st.plotly_chart(fig, use_container_width=True)
        
        # Model accuracy metrics
        if 'predicted_downlink_throughput' in filtered_df.columns and 'DL_bitrate' in filtered_df.columns:  # Changed from y_downlink_throughput
            st.markdown("#### 📊 Model Accuracy Statistics")
            
            # Calculate accuracy metrics
            actual = filtered_df['DL_bitrate'].dropna()  # Changed from y_downlink_throughput
            predicted = filtered_df['predicted_downlink_throughput'].dropna()
            
            if len(actual) > 0 and len(predicted) > 0:
                min_len = min(len(actual), len(predicted))
                actual = actual.iloc[:min_len]
                predicted = predicted.iloc[:min_len]
                
                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Mean Absolute Error", f"{mae:.2f} Mbps")
                with metric_col2:
                    st.metric("Root Mean Square Error", f"{rmse:.2f} Mbps")
                with metric_col3:
                    st.metric("Mean Absolute Percentage Error", f"{mape:.1f}%")
    
    with pred_tab3:
        st.markdown("#### AI-Powered Anomaly Detection & Risk Assessment")
        
        if 'anomaly_proba' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # High-risk nodes
                high_risk_threshold = filtered_df['anomaly_proba'].quantile(0.8)
                high_risk_nodes = filtered_df[filtered_df['anomaly_proba'] > high_risk_threshold]
                
                st.markdown(f"**🔥 High-Risk Nodes ({len(high_risk_nodes)} detected)**")
                if not high_risk_nodes.empty:
                    risk_summary = high_risk_nodes.groupby('Node').agg({
                        'anomaly_proba': 'mean',
                        'risk_score': 'mean' if 'risk_score' in filtered_df.columns else lambda x: 0
                    }).reset_index()
                    risk_summary.columns = ['Node', 'Anomaly Probability', 'Risk Score']
                    st.dataframe(risk_summary.head(10), use_container_width=True)
            
            with col2:
                # Anomaly probability distribution
                fig = px.histogram(filtered_df, x='anomaly_proba',
                                 title="Anomaly Probability Distribution",
                                 labels={'anomaly_proba': 'Anomaly Probability'})
                fig.add_vline(x=high_risk_threshold, line_dash="dash", line_color="red",
                            annotation_text="High Risk Threshold")
                st.plotly_chart(fig, use_container_width=True)
        
        # Proactive alerts
        if 'risk_score' in filtered_df.columns:
            st.markdown("#### 🚨 Proactive Network Alerts")
            
            critical_nodes = filtered_df[filtered_df['risk_score'] > 0.7]
            warning_nodes = filtered_df[(filtered_df['risk_score'] > 0.4) & (filtered_df['risk_score'] <= 0.7)]
            
            if not critical_nodes.empty:
                st.markdown(f"<div class='alert-box danger-alert'><b>🔴 CRITICAL: {len(critical_nodes)} nodes require immediate attention</b></div>", 
                           unsafe_allow_html=True)
                
            if not warning_nodes.empty:
                st.markdown(f"<div class='alert-box warning-alert'><b>🟡 WARNING: {len(warning_nodes)} nodes show degrading performance</b></div>", 
                           unsafe_allow_html=True)
            
            if critical_nodes.empty and warning_nodes.empty:
                st.markdown("<div class='alert-box success-alert'><b>✅ All nodes operating within normal parameters</b></div>", 
                           unsafe_allow_html=True)
    
    with pred_tab4:
        st.markdown("#### 🤖 AI-Driven Network Optimization Recommendations")
        
        # Generate intelligent recommendations based on predictions
        recommendations = []
        
        if 'predicted_downlink_throughput' in filtered_df.columns:
            low_throughput_nodes = filtered_df[filtered_df['predicted_downlink_throughput'] < filtered_df['predicted_downlink_throughput'].quantile(0.2)]
            if not low_throughput_nodes.empty:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Throughput Optimization',
                    'description': f'Optimize {len(low_throughput_nodes.Node.unique())} nodes with predicted low throughput',
                    'action': 'Review antenna configuration, check for interference, consider carrier aggregation'
                })
        
        if 'risk_score' in filtered_df.columns:
            high_risk_nodes = filtered_df[filtered_df['risk_score'] > 0.6]
            if not high_risk_nodes.empty:
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Risk Mitigation',
                    'description': f'Immediate intervention required for {len(high_risk_nodes.Node.unique())} high-risk nodes',
                    'action': 'Schedule maintenance, backup traffic routing, monitor closely'
                })
        
        if 'anomaly_proba' in filtered_df.columns:
            anomaly_nodes = filtered_df[filtered_df['anomaly_proba'] > 0.5]
            if not anomaly_nodes.empty:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Anomaly Prevention',
                    'description': f'Proactive maintenance for {len(anomaly_nodes.Node.unique())} nodes showing anomaly patterns',
                    'action': 'Predictive maintenance scheduling, parameter optimization'
                })
        
        # Network technology optimization
        if 'NetworkTech' in filtered_df.columns and 'predicted_downlink_throughput' in filtered_df.columns:
            tech_performance = filtered_df.groupby('NetworkTech')['predicted_downlink_throughput'].mean()
            underperforming_tech = tech_performance[tech_performance < tech_performance.median()].index
            if len(underperforming_tech) > 0:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Technology Upgrade',
                    'description': f'Consider upgrading {", ".join(underperforming_tech)} technology',
                    'action': 'Evaluate 5G deployment, optimize existing technology parameters'
                })
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}
                color = priority_color.get(rec['priority'], '#6c757d')
                
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 1rem; margin: 1rem 0; background-color: rgba(0,0,0,0.05); border-radius: 5px;">
                    <h4 style="color: {color}; margin: 0;">🎯 {rec['category']} - {rec['priority']} Priority</h4>
                    <p style="margin: 0.5rem 0;"><b>Issue:</b> {rec['description']}</p>
                    <p style="margin: 0; font-style: italic;"><b>Recommended Action:</b> {rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("✅ **No immediate optimization recommendations at this time. Network performance is within expected parameters.**")
        
        # Future predictions summary
        if 'predicted_downlink_throughput' in filtered_df.columns:
            st.markdown("#### 📅 Short-term Network Performance Forecast")
            
            # Calculate trend
            recent_data = filtered_df.tail(24)  # Last 24 hours
            if len(recent_data) > 1:
                trend_dl = (recent_data['predicted_downlink_throughput'].iloc[-1] - recent_data['predicted_downlink_throughput'].iloc[0]) / len(recent_data)
                trend_direction = "📈 Improving" if trend_dl > 0 else "📉 Declining" if trend_dl < 0 else "➡️ Stable"
                
                st.info(f"""
                **Network Performance Trend:** {trend_direction}
                
                **Expected Changes in Next 24 Hours:**
                - Downlink throughput trend: {trend_dl:+.2f} Mbps per hour
                - Recommended monitoring frequency: {'High (every 15 min)' if abs(trend_dl) > 1 else 'Normal (hourly)'}
                """)
    
    # Additional analysis
    st.markdown("### 📊 Advanced Analytics")
    
    col1, col2 = st.columns(2)

    with col1:
        if 'NetworkTech' in filtered_df.columns and 'DL_bitrate' in filtered_df.columns:  # Changed from y_downlink_throughput
            fig = px.box(filtered_df, x='NetworkTech', y='DL_bitrate',  # Changed from y_downlink_throughput
                        title="Downlink Throughput by Network Technology",
                        labels={'DL_bitrate': 'Throughput (Mbps)'})  # Changed from y_downlink_throughput
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Mobility' in filtered_df.columns and 'PINGAVG' in filtered_df.columns:  # Changed from y_latency
            fig = px.violin(filtered_df, x='Mobility', y='PINGAVG',  # Changed from y_latency
                           title="Latency Distribution by Mobility Type",
                           labels={'PINGAVG': 'Latency (ms)'})  # Changed from y_latency
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
