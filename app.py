import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 웹페이지 기본 설정 (TH Chart 브랜딩)
st.set_page_config(page_title="TH Chart | AI Dashboard", page_icon="👑", layout="wide")

# 커스텀 CSS 적용 (바이낸스 다크 테마)
st.markdown("""
    <style>
    /* 전체 배경 및 텍스트 색상 (바이낸스 스타일) */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0e11;
        color: #EAECEF;
    }
    /* 사이드바 배경색 */
    [data-testid="stSidebar"] {
        background-color: #181a20;
    }
    /* 상단 헤더 투명화 */
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    /* 타이틀 및 서브타이틀 */
    .main-title {
        font-size: 50px;
        font-weight: 900;
        color: #F3BA2F; /* 바이낸스 골드 포인트 */
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px #000000;
    }
    .sub-title {
        font-size: 20px;
        color: #848E9C;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    /* 하단 메트릭(수치) 박스 스타일링 */
    div[data-testid="metric-container"] {
        background-color: #181a20;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2B3139;
    }
    /* 알림 박스 커스텀 */
    .stAlert {
        background-color: #181a20 !important;
        border: 1px solid #2B3139 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">👑 TH Chart</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Taeha\'s AI-Powered BTC 15m Prediction Dashboard</p>', unsafe_allow_html=True)

# 사이드바 프로필
st.sidebar.title("✨ TH Chart Info")
st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 **CEO / Developer:** 태하 (Taeha)")
st.sidebar.success("🤖 **Core Engine:** XGBoost MTF")

# 자동 새로고침 및 기간 설정
auto_refresh = st.sidebar.checkbox("⚡ **1분 자동 새로고침 켜기**", value=True)
# yfinance 15m 데이터 최대 제공 기간인 60일(여유분 2일 제외 58일)로 확장. 기본값을 1일로 변경 (모바일 최적화)
chart_days = st.sidebar.slider("📊 차트 표시 기간 (일)", min_value=1, max_value=58, value=1, help="차트에 표시할 과거 예측 기록의 기간을 선택하세요. (최대 58일)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Signal Guide")
st.sidebar.write("- <span style='color:#0ECB81'>▲</span> **Pred Up:** 상승 돌파 예상 (Long)", unsafe_allow_html=True)
st.sidebar.write("- <span style='color:#F6465D'>▼</span> **Pred Down:** 하락 이탈 예상 (Short)", unsafe_allow_html=True)
st.sidebar.write("*AI 시그널의 연속성을 확인하고 진입하세요.*")

LOG_FILE = "prediction_history.csv"

def update_prediction_log(latest_time, latest_close, latest_pred):
    # 기록 파일이 없으면 생성
    if not os.path.exists(LOG_FILE):
        df_log = pd.DataFrame(columns=["Time", "Close Price", "Prediction"])
        df_log.to_csv(LOG_FILE, index=False)

    df_log = pd.read_csv(LOG_FILE)
    latest_time_str = str(latest_time)

    # 새로운 시간의 캔들일 경우에만 기록 추가
    if len(df_log) == 0 or df_log.iloc[-1]["Time"] != latest_time_str:
        pred_text = "상승 돌파 📈" if latest_pred == 2 else ("하락 이탈 📉" if latest_pred == 0 else "횡보 관망 ⏳")
        new_row = pd.DataFrame([{"Time": latest_time_str, "Close Price": f"${latest_close:,.2f}", "Prediction": pred_text}])
        df_log = pd.concat([df_log, new_row], ignore_index=True)
        df_log.to_csv(LOG_FILE, index=False)

    # 최근 15개 기록만 반환
    return df_log.tail(15)

def get_live_chart(days_to_show):
    fetch_days = min(days_to_show + 2, 60) # 이동평균선 계산을 위해 더 가져오되 최대 60일 제한
    btc_df = yf.Ticker('BTC-USD').history(interval='15m', period=f'{fetch_days}d')
    btc_df = btc_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    btc_df.index = btc_df.index.tz_convert('Asia/Seoul').tz_localize(None)

    # --- ATR(Average True Range) 계산 로직 추가 --- #
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

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_7', 'RSI_14', 'SMA_1H', 'SMA_4H', 'Vol_4H', 'SMA_24H', 'BB_Width']
    X_live = btc_df[features].copy()

    # 모델 경로 개선 (폴더 구조가 깨져도 루트에서 찾을 수 있도록 폴백 추가)
    model_path = './data/model/xgboost_btc_15m_3class_strict.pkl'
    if not os.path.exists(model_path):
        model_path = 'xgboost_btc_15m_3class_strict.pkl'
    model_xgb = joblib.load(model_path)

    X_live['Pred'] = model_xgb.predict(X_live[features])

    data_points = min(len(X_live), days_to_show * 96) # 선택한 일수만큼 캔들 표시
    recent_eval = X_live.iloc[-data_points:]

    # 바이낸스 컬러: 상승 #0ECB81, 하락 #F6465D
    fig = go.Figure(data=[go.Candlestick(x=recent_eval.index,
                    open=recent_eval['Open'], high=recent_eval['High'],
                    low=recent_eval['Low'], close=recent_eval['Close'],
                    increasing_line_color='#0ECB81', decreasing_line_color='#F6465D',
                    name='BTC/USD', showlegend=False)]) # 공간 절약을 위해 기본 범례 숨김

    pred_up = recent_eval[recent_eval['Pred'] == 2]
    pred_down = recent_eval[recent_eval['Pred'] == 0]

    fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up['Low'] * 0.998,
                             mode='markers', marker=dict(symbol='triangle-up', size=16, color='#0ECB81', line=dict(width=1.5, color='white')),
                             name='🟢 Pred Up'))
    fig.add_trace(go.Scatter(x=pred_down.index, y=pred_down['High'] * 1.002,
                             mode='markers', marker=dict(symbol='triangle-down', size=16, color='#F6465D', line=dict(width=1.5, color='white')),
                             name='🔴 Pred Down'))

    # 모바일 최적화: 레이아웃 여백 최소화, 높이 축소, 범례 가로 배치, 드래그 이동 설정
    time_str = recent_eval.index[-1].strftime("%Y-%m-%d %H:%M")
    fig.update_layout(
        title=dict(text=f'<b>TH Chart Live Tracking</b><br><span style="font-size:12px;color:gray;">Updated: {time_str}</span>', font=dict(size=16, color='#EAECEF')),
        yaxis_title='BTC Price (USD)',
        xaxis_title='', # 공간 절약을 위해 x축 라벨 제거
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=450, # 모바일에 맞는 높이 유지
        margin=dict(l=10, r=10, t=55, b=45), # 좌우 여백 최소화, 아래 여백 확보
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uirevision='live_chart',
        dragmode='pan', # 기본 동작을 '이동(pan)'으로 변경
        legend=dict( # 범례를 차트 가로 하단으로 이동하여 폭 확보
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    latest_preds = X_live['Pred'].iloc[-5:].values
    latest_close = recent_eval['Close'].iloc[-1]
    latest_atr = btc_df['ATR_14'].iloc[-1] # 방금 계산된 최신 ATR 값 추출
    return fig, latest_preds, X_live.index[-1], latest_close, latest_atr

try:
    fig, latest_preds, last_time, last_close, latest_atr = get_live_chart(chart_days)

    # 스크롤 줌(마우스 휠/핀치) 활성화 및 불필요한 툴바 숨기기
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    st.markdown("### 📊 실시간 AI 브리핑 패널")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🔄 마지막 캔들 시간", value=str(last_time))
    with col2:
        pred_text = "상승 돌파 📈" if latest_preds[-1] == 2 else ("하락 이탈 📉" if latest_preds[-1] == 0 else "횡보 관망 ⏳")
        st.metric(label="🎯 현재 캔들 예측", value=pred_text)
    with col3:
        st.metric(label="💲 현재가 (Close)", value=f"${last_close:,.2f}")

    # --- ATR 기반 손익절 타겟 UI --- #
    if latest_preds[-1] == 2:
        sl_price = last_close - (1.5 * latest_atr)
        tp_price = last_close + (3.0 * latest_atr)
        st.success(f"🟢 **[Long 포지션 진입 시그널]** 익절가(TP): **${tp_price:,.2f}** 🎯 | 손절가(SL): **${sl_price:,.2f}** 🛡️ (1:2 ATR 전략)")
    elif latest_preds[-1] == 0:
        sl_price = last_close + (1.5 * latest_atr)
        tp_price = last_close - (3.0 * latest_atr)
        st.error(f"🔴 **[Short 포지션 진입 시그널]** 익절가(TP): **${tp_price:,.2f}** 🎯 | 손절가(SL): **${sl_price:,.2f}** 🛡️ (1:2 ATR 전략)")
    else:
        st.info("⏳ **[관망 시그널]** 현재는 시장의 방향성이 불명확하여 포지션 진입을 권장하지 않습니다.")

    st.markdown("---")
    col_log1, col_log2 = st.columns([4, 1])
    with col_log1:
        st.markdown("### 📝 AI 예측 자동 기록부 (최근 15개)")
    with col_log2:
        if os.path.exists(LOG_FILE):
            full_df = pd.read_csv(LOG_FILE)
            csv_data = full_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 전체 기록 다운로드", data=csv_data, file_name="TH_Chart_Prediction_History.csv", mime="text/csv")

    history_df = update_prediction_log(last_time, last_close, latest_preds[-1])
    # 최신 기록이 맨 위로 오도록 역순 출력
    st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"오류 발생: {e}")

if auto_refresh:
    time.sleep(60)
    st.rerun()
