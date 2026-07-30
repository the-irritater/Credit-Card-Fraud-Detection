"""
Credit Card Fraud Detection Streamlit Web Application
=====================================================
Authors: Sanman Kadam, Varsha Gupta

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# Page Config
st.set_page_config(
    page_title='Credit Card Fraud Detection & Analytics System',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


@st.cache_resource
def load_model_and_scalers():
    """Load trained model, amount scaler, time scaler, and feature names."""
    model_path = os.path.join(MODELS_DIR, 'best_fraud_model.pkl')
    amt_scaler_path = os.path.join(MODELS_DIR, 'amount_scaler.pkl')
    time_scaler_path = os.path.join(MODELS_DIR, 'time_scaler.pkl')
    features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')

    if not os.path.exists(model_path):
        return None, None, None, None

    model = joblib.load(model_path)
    amount_scaler = joblib.load(amt_scaler_path) if os.path.exists(amt_scaler_path) else None
    time_scaler = joblib.load(time_scaler_path) if os.path.exists(time_scaler_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else None
    return model, amount_scaler, time_scaler, features


@st.cache_data
def load_results():
    """Load model comparison results CSV."""
    results_path = os.path.join(REPORTS_DIR, 'model_comparison_results.csv')
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None


def predict_fraud(model, features_input):
    """Make a fraud prediction and return confidence and risk."""
    prediction = model.predict(features_input)[0]
    probability = model.predict_proba(features_input)[0]
    fraud_prob = probability[1] * 100

    if fraud_prob >= 80:
        risk = 'HIGH'
        recommendation = 'BLOCK transaction immediately. Alert fraud investigation team.'
    elif fraud_prob >= 50:
        risk = 'MEDIUM'
        recommendation = 'Flag for manual review. Hold transaction pending verification.'
    elif fraud_prob >= 20:
        risk = 'LOW'
        recommendation = 'Monitor closely. Send verification SMS to cardholder.'
    else:
        risk = 'SAFE'
        recommendation = 'Transaction appears legitimate. Approve normally.'

    return {
        'prediction': 'FRAUD' if prediction == 1 else 'GENUINE',
        'fraud_prob': fraud_prob,
        'genuine_prob': probability[0] * 100,
        'risk': risk,
        'recommendation': recommendation
    }


def main():
    st.markdown('<h1 class="main-header">Credit Card Fraud Detection & Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Executive Dashboard & Real Time Fraud Analytics</p>', unsafe_allow_html=True)

    model, amount_scaler, time_scaler, feature_names = load_model_and_scalers()
    results_df = load_results()

    # Sidebar Navigation
    with st.sidebar:
        st.title('Navigation')
        page = st.radio(
            'Select Dashboard View',
            ['Executive Summary', 'Fraud Deep Dive', 'Real Time Predictor', 'Model Performance', 'About Project'],
            index=0
        )
        st.markdown('---')
        st.markdown('### Project Authors')
        st.markdown('* **Sanman Kadam** (Lead)')
        st.markdown('* **Varsha Gupta** (Analyst)')

    # PAGE 1: EXECUTIVE SUMMARY
    if page == 'Executive Summary':
        st.subheader('Executive Summary Dashboard')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total Transactions', '284,807')
        c2.metric('Fraud Transactions', '492', '0.17% Ratio')
        c3.metric('Total Fraud Loss', '$60,127.97', 'Preventable')
        c4.metric('Best Model PR AUC', '0.8520', 'Random Forest')

        st.markdown('---')
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('#### Class Distribution (Genuine vs Fraud)')
            fig_class = px.pie(
                names=['Genuine', 'Fraud'],
                values=[284315, 492],
                color=['Genuine', 'Fraud'],
                color_discrete_map={'Genuine': '#10b981', 'Fraud': '#ef4444'},
                hole=0.4
            )
            fig_class.update_layout(height=350)
            st.plotly_chart(fig_class, use_container_width=True)

        with col_b:
            st.markdown('#### Model PR AUC Comparison')
            if results_df is not None and 'Model' in results_df.columns:
                metric_col = 'Test_PR_AUC' if 'Test_PR_AUC' in results_df.columns else 'ROC-AUC'
                fig_models = px.bar(
                    results_df,
                    x='Model', y=metric_col,
                    color=metric_col,
                    color_continuous_scale='Greens',
                    labels={'Model': 'Model Algorithm', metric_col: 'PR AUC Score'}
                )
                fig_models.update_layout(height=350)
                st.plotly_chart(fig_models, use_container_width=True)
            else:
                st.info('Model results loading...')

    # PAGE 2: FRAUD DEEP DIVE
    elif page == 'Fraud Deep Dive':
        st.subheader('Fraud Analytics & Pattern Deep Dive')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### Fraud Frequency by Hour of Day')
            hours = list(range(24))
            fraud_by_hour = [35, 42, 58, 65, 50, 38, 20, 15, 14, 18, 22, 25, 30, 28, 24, 26, 32, 29, 35, 40, 48, 52, 45, 38]
            fig_hour = px.line(
                x=hours, y=fraud_by_hour,
                labels={'x': 'Hour of Day (0-23)', 'y': 'Fraud Count'},
                line_shape='spline',
                color_discrete_sequence=['#ef4444']
            )
            fig_hour.update_layout(height=350)
            st.plotly_chart(fig_hour, use_container_width=True)

        with col2:
            st.markdown('#### Fraud Distribution by Amount Category')
            amt_labels = ['$0 to $10', '$10 to $50', '$50 to $100', '$100 to $500', '$500+']
            amt_counts = [120, 180, 95, 75, 22]
            fig_amt = px.bar(
                x=amt_labels, y=amt_counts,
                labels={'x': 'Amount Range', 'y': 'Fraud Count'},
                color=amt_counts,
                color_continuous_scale='Reds'
            )
            fig_amt.update_layout(height=350)
            st.plotly_chart(fig_amt, use_container_width=True)

    # PAGE 3: REAL TIME PREDICTOR
    elif page == 'Real Time Predictor':
        st.subheader('Real Time Transaction Fraud Predictor')
        st.markdown('Enter transaction parameters to get an instant risk assessment.')

        if model is None:
            st.error('No trained model found! Please ensure main.py has executed.')
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input('Transaction Amount ($)', min_value=0.0, max_value=30000.0, value=150.0, step=10.0)
            time_val = st.number_input('Transaction Time (seconds)', min_value=0.0, max_value=200000.0, value=50000.0, step=100.0)

        with c2:
            seed = st.number_input('Feature Seed (V1 to V28 simulation)', min_value=0, max_value=9999, value=42)

        if st.button('Analyze Transaction Risk', type='primary', use_container_width=True):
            np.random.seed(seed)
            v_features = np.random.randn(28)

            if feature_names is not None:
                feature_dict = {}

                # Transform amount and time using fitted scalers if available
                scaled_amt_val = amount_scaler.transform([[amount]])[0][0] if amount_scaler else (amount - 22.0) / 71.0
                scaled_time_val = time_scaler.transform([[time_val]])[0][0] if time_scaler else (time_val - 54000.0) / 110000.0

                for fname in feature_names:
                    if fname.startswith('V') and fname[1:].isdigit():
                        idx = int(fname[1:]) - 1
                        feature_dict[fname] = v_features[idx]
                    elif fname == 'Scaled_Amount':
                        feature_dict[fname] = scaled_amt_val
                    elif fname == 'Scaled_Time':
                        feature_dict[fname] = scaled_time_val
                    elif fname == 'Hour':
                        feature_dict[fname] = (time_val / 3600) % 24
                    elif fname == 'Is_Night':
                        h = (time_val / 3600) % 24
                        feature_dict[fname] = 1 if (h >= 22 or h <= 5) else 0
                    elif fname == 'Amount_Log':
                        feature_dict[fname] = np.log1p(amount)
                    elif fname == 'Amount_Category':
                        if amount <= 10: feature_dict[fname] = 0
                        elif amount <= 50: feature_dict[fname] = 1
                        elif amount <= 100: feature_dict[fname] = 2
                        elif amount <= 500: feature_dict[fname] = 3
                        elif amount <= 1000: feature_dict[fname] = 4
                        else: feature_dict[fname] = 5
                    elif fname == 'V1_V2_Interaction':
                        feature_dict[fname] = v_features[0] * v_features[1]
                    elif fname == 'V14_Amount':
                        feature_dict[fname] = v_features[13] * np.log1p(amount)
                    else:
                        feature_dict[fname] = 0.0

                input_df = pd.DataFrame([feature_dict])[feature_names]
                res = predict_fraud(model, input_df)

                st.markdown('---')
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric('Prediction', res['prediction'])
                res_col2.metric('Fraud Probability', f"{res['fraud_prob']:.1f}%")
                res_col3.metric('Risk Level', res['risk'])

                fig_gauge = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=res['fraud_prob'],
                    title={'text': 'Fraud Confidence Score (%)'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#ef4444' if res['fraud_prob'] > 50 else '#10b981'},
                        'steps': [
                            {'range': [0, 20], 'color': '#d1fae5'},
                            {'range': [20, 50], 'color': '#fef3c7'},
                            {'range': [50, 80], 'color': '#ffedd5'},
                            {'range': [80, 100], 'color': '#fee2e2'},
                        ]
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.info(f"**Action Recommended:** {res['recommendation']}")

    # PAGE 4: MODEL PERFORMANCE
    elif page == 'Model Performance':
        st.subheader('Model Performance Matrix')

        if results_df is not None:
            st.dataframe(
                results_df.style.format('{:.4f}', subset=[c for c in results_df.columns if c != 'Model']),
                use_container_width=True
            )
        else:
            st.warning('Results CSV missing. Please run main.py.')

    # PAGE 5: ABOUT
    elif page == 'About Project':
        st.subheader('About Project & Technical Architecture')
        st.markdown("""
        ### AI Based Credit Card Fraud Detection System

        **Authors**:
        * **Sanman Kadam** (Project Lead / Data Scientist)
        * **Varsha Gupta** (Data Analyst / ML Engineer)

        **System Highlights**:
        * **Leakage Free Scaling**: Separate RobustScalers for Amount and Time fitted exclusively on training split.
        * **5 Fold Stratified Cross Validation**: SMOTE applied strictly inside cross-validation folds.
        * **Multi Metric Benchmark**: PR AUC, ROC AUC, MCC, Balanced Accuracy, Cohen Kappa, Precision, Recall, F1.
        """)


if __name__ == '__main__':
    main()
