import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import warnings
import datetime
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings('ignore')

# 웹페이지 기본 설정
st.set_page_config(page_title="TH Chart | Pro Dashboard", page_icon="⚡", layout="wide")

# 전문 Auto-Refresh 엔진: 1분(60000ms)마다 브라우저 자동 새로고침
st_autorefresh(interval=60000, limit=None, key="auto_refresh_timer")

# 커스텀 CSS 적용 (프리미엄 테마)
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
st.markdown('<p class="sub-title">Live Kraken Engine (Real-Time Anti-Freeze)</p>', unsafe_allow_html=True)

# 서버 시간 계산
now_utc = datetime.datetime.utcnow()
now_kst = now_utc + datetime.timedelta(hours=9)

st.sidebar.title("⚙️ System Status")
st.sidebar.markdown("---")
st.sidebar.success("🟢 **Live Engine:** Active (Kraken)")
st.sidebar.info("🔄 **Auto Refresh:** 60s")
st.sidebar.markdown("**App Version:** `V6_KRAKEN_ENGINE`")
st.sidebar.markdown(f"**서버 시간 (UTC):** `{now_utc.strftime('%H:%M:%S')}`")
st.sidebar.markdown(f"**서버 시간 (KST):** `{now_kst.strftime('%H:%M:%S')}`")

chart_days = st.sidebar.slider("📊 차트 표시 기간 (일)", min_value=1, max_value=30, value=1)

# 바이낸스 차단 우회, yfinance 지연 해결을 위한 크라켄(Kraken) 거래소 사용
def fetch_and_process_data(days_to_show):
    fetch_days = min(days_to_show + 2, 60)
    limit = fetch_days * 96

    # 미국 IP 차단이 없는 Kraken 거래소 실시간 데이터 호출
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=limit)

    if not ohlcv:
        raise ValueError("크라켄 거래소에서 데이터를 가져오지 못했습니다.")

    df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df.set_index('Datetime', inplace=True)

    # 원본 UTC 타임을 KST로 변환
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
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

    try:
        model_xgb = joblib.load(model_path)
    except Exception as e:
        st.error("AI 모델 파일을 불러오지 못했습니다. github에 pkl 파일이 누락되었을 수 있습니다.")
        return None, None, btc_df.index[-1], btc_df['Close'].iloc[-1], btc_df['ATR_14'].iloc[-1]

    X_live['Pred'] = model_xgb.predict(X_live[features])

    data_points = min(len(X_live), days_to_show * 96)
    recent_eval = X_live.iloc[-data_points:]

    fig = go.Figure(data=[go.Candlestick(x=recent_eval.index,
                    open=recent_eval['Open'], high=recent_eval['High'],
                    low=recent_eval['Low'], close=recent_eval['Close'],
                    increasing_line_color='#00E676', decreasing_line_color='#FF3D00',
                    name='BTC/USDT', showlegend=False)])

    pred_up = recent_eval[recent_eval['Pred'] == 2]
    pred_down = recent_eval[recent_eval['Pred'] == 0]

    fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up['Low'] * 0.998,
                             mode='markers', marker=dict(symbol='triangle-up', size=18, color='#00E676', line=dict(width=2, color='white')),
                             name='🟢 Long Target'))
    fig.add_trace(go.Scatter(x=pred_down.index, y=pred_down['High'] * 1.002,
                             mode='markers', marker=dict(symbol='triangle-down', size=18, color='#FF3D00', line=dict(width=2, color='white')),
                             name='🔴 Short Target'))

    time_str = recent_eval.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    fig.update_layout(
        title=dict(text=f'<b>Live Tracker</b> (Last Kraken Candle: {time_str})', font=dict(size=18, color='#EAECEF')),
        yaxis_title='USDT',
        xaxis_title='',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uirevision='live_chart',
        dragmode='pan',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    latest_preds = X_live['Pred'].iloc[-5:].values
    latest_close = recent_eval['Close'].iloc[-1]
    latest_atr = btc_df['ATR_14'].iloc[-1]
    return fig, latest_preds, X_live.index[-1], latest_close, latest_atr

try:
    fig, latest_preds, last_time, last_close, latest_atr = get_live_chart(chart_days)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

        st.markdown("### 🤖 AI Core Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="⏳ 최신 캔들 타임스탬프 (KST)", value=str(last_time))
        with col2:
            pred_text = "상승 돌파 📈" if latest_preds[-1] == 2 else ("하락 이탈 📉" if latest_preds[-1] == 0 else "방향성 모호 ⏳")
            st.metric(label="🎯 AI 최종 예측", value=pred_text)
        with col3:
            st.metric(label="💵 현재 가격", value=f"${last_close:,.2f}")

        if latest_preds[-1] == 2:
            st.success(f"🟢 **[LONG 시그널]** 추천 익절가: **${last_close + (3.0 * latest_atr):,.2f}** | 추천 손절가: **${last_close - (1.5 * latest_atr):,.2f}**")
        elif latest_preds[-1] == 0:
            st.error(f"🔴 **[SHORT 시그널]** 추천 익절가: **${last_close - (3.0 * latest_atr):,.2f}** | 추천 손절가: **${last_close + (1.5 * latest_atr):,.2f}**")
        else:
            st.info("⏳ **[WAIT]** 현재는 시장의 변동성이 부족하거나 방향이 불명확합니다.")

except Exception as e:
    st.error(f"🚨 시스템 오류 발생 (자동으로 재시도합니다): {e}")
