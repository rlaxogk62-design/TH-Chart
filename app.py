import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import warnings
import datetime
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings('ignore')

st.set_page_config(page_title="TH Chart | Pro Dashboard", page_icon="⚡", layout="wide")
st_autorefresh(interval=60000, limit=None, key="auto_refresh_timer")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #EAECEF; }
    [data-testid="stSidebar"] { background-color: #161A25; }
    [data-testid="stHeader"] { background-color: transparent; }
    .main-title { font-size: 55px; font-weight: 900; color: #00E676; text-align: center; margin-bottom: 0px; text-shadow: 0px 0px 10px rgba(0,230,118,0.5); }
    .sub-title { font-size: 22px; color: #A0AEC0; text-align: center; margin-top: -10px; margin-bottom: 30px; }
    div[data-testid="metric-container"] { background-color: #1A202C; border-radius: 10px; padding: 20px; border: 1px solid #2D3748; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ TH Chart Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Live Yahoo Finance Engine</p>', unsafe_allow_html=True)

now_utc = datetime.datetime.utcnow()
now_kst = now_utc + datetime.timedelta(hours=9)

st.sidebar.title("⚙️ System Status")
st.sidebar.markdown("---")
st.sidebar.success("🟢 **Live Engine:** Active (Yahoo Finance)")
st.sidebar.info("🔄 **Auto Refresh:** 60s")
st.sidebar.markdown(f"**서버 시간 (KST):** `{now_kst.strftime('%H:%M:%S')}`")

chart_days = st.sidebar.slider("📊 차트 표시 기간 (일)", min_value=1, max_value=30, value=1)

def fetch_and_process_data(days_to_show):
    fetch_days = min(days_to_show + 2, 60)
    df = yf.Ticker('BTC-USD').history(interval='15m', period=f'{fetch_days}d')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
    else:
        df.index = df.index.tz_convert('Asia/Seoul').tz_localize(None)
    
    btc_df = df
    btc_df['Prev_Close'] = btc_df['Close'].shift(1)
    tr1 = btc_df['High'] - btc_df['Low']
    tr2 = (btc_df['High'] - btc_df['Prev_Close']).abs()
    tr3 = (btc_df['Low'] - btc_df['Prev_Close']).abs()
    btc_df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    btc_df['ATR_14'] = btc_df['TR'].rolling(window=14).mean()

    btc_df['Returns'] = btc_df['Close'].pct_change()
    btc_df['SMA_7'] = btc_df['Close'].rolling(window=7).mean()

    delta = btc_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    btc_df['RSI_14'] = 100 - (100 / (1 + gain / loss))

    btc_df['SMA_1H'] = btc_df['Close'].rolling(window=4).mean()
    btc_df['SMA_4H'] = btc_df['Close'].rolling(window=16).mean()
    btc_df['Vol_4H'] = btc_df['Returns'].rolling(window=16).std()
    btc_df['SMA_24H'] = btc_df['Close'].rolling(window=96).mean()

    btc_df['BB_Std'] = btc_df['Close'].rolling(window=20).std()
    btc_df['BB_Width'] = (btc_df['BB_Std'] * 4) / btc_df['Close'].rolling(window=20).mean()

    btc_df.dropna(inplace=True)
    return btc_df

def get_live_chart(days_to_show):
    btc_df = fetch_and_process_data(days_to_show)
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_7', 'RSI_14', 'SMA_1H', 'SMA_4H', 'Vol_4H', 'SMA_24H', 'BB_Width']
    X_live = btc_df[features].copy()

    model_path = './data/model/xgboost_btc_15m_3class_strict.pkl'
    if not os.path.exists(model_path):
        model_path = 'xgboost_btc_15m_3class_strict.pkl'

    model_xgb = joblib.load(model_path)
    X_live['Pred'] = model_xgb.predict(X_live[features])

    data_points = min(len(X_live), days_to_show * 96)
    recent_eval = X_live.iloc[-data_points:]

    fig = go.Figure(data=[go.Candlestick(x=recent_eval.index,
                    open=recent_eval['Open'], high=recent_eval['High'],
                    low=recent_eval['Low'], close=recent_eval['Close'],
                    increasing_line_color='#00E676', decreasing_line_color='#FF3D00')])

    pred_up = recent_eval[recent_eval['Pred'] == 2]
    pred_down = recent_eval[recent_eval['Pred'] == 0]

    fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up['Low'] * 0.998,
                             mode='markers', marker=dict(symbol='triangle-up', size=18, color='#00E676', line=dict(width=2, color='white'))))
    fig.add_trace(go.Scatter(x=pred_down.index, y=pred_down['High'] * 1.002,
                             mode='markers', marker=dict(symbol='triangle-down', size=18, color='#FF3D00', line=dict(width=2, color='white'))))

    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig, X_live['Pred'].iloc[-5:].values, X_live.index[-1], recent_eval['Close'].iloc[-1], btc_df['ATR_14'].iloc[-1]

try:
    fig, latest_preds, last_time, last_close, latest_atr = get_live_chart(chart_days)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🤖 AI Core Analysis")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="⏳ 최신 캔들 타임스탬프 (KST)", value=str(last_time))
    with col2: st.metric(label="🎯 AI 최종 예측", value="상승 돌파 📈" if latest_preds[-1] == 2 else ("하락 이탈 📉" if latest_preds[-1] == 0 else "방향성 모호 ⏳"))
    with col3: st.metric(label="💵 현재 가격", value=f"${last_close:,.2f}")
except Exception as e:
    st.error(f"🚨 오류: {e}")
