import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="글로벌 주식 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 종목 리스트 (주요 글로벌 주식/지수)
# ---------------------------------------------------------
TICKERS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Samsung Electronics (005930.KS)": "005930.KS",
    "SK Hynix (000660.KS)": "000660.KS",
    "Toyota (7203.T)": "7203.T",
    "TSMC (TSM)": "TSM",
    "Alibaba (BABA)": "BABA",
    "S&P 500 지수 (^GSPC)": "^GSPC",
    "나스닥 지수 (^IXIC)": "^IXIC",
    "다우존스 지수 (^DJI)": "^DJI",
    "코스피 지수 (^KS11)": "^KS11",
    "닛케이225 지수 (^N225)": "^N225",
}

PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "5년": "5y",
    "연초 이후": "ytd",
    "전체": "max",
}

INTERVAL_OPTIONS = {
    "1일": "1d",
    "1주": "1wk",
    "1달": "1mo",
}


# ---------------------------------------------------------
# 데이터 로딩 함수 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_price_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data is not None and not data.empty:
        data.index = pd.to_datetime(data.index)
    return data


@st.cache_data(ttl=300)
def load_ticker_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_multi_summary(ticker_map: dict) -> pd.DataFrame:
    rows = []
    for name, symbol in ticker_map.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if hist.empty or len(hist) < 2:
                continue
            last_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2]
            change = last_close - prev_close
            pct_change = (change / prev_close) * 100
            rows.append(
                {
                    "종목명": name,
                    "티커": symbol,
                    "현재가": round(last_close, 2),
                    "전일대비": round(change, 2),
                    "등락률(%)": round(pct_change, 2),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 사이드바 - 사용자 입력
# ---------------------------------------------------------
st.sidebar.title("📊 설정")

selected_name = st.sidebar.selectbox("종목 선택", list(TICKERS.keys()), index=0)
selected_ticker = TICKERS[selected_name]

custom_ticker = st.sidebar.text_input(
    "직접 티커 입력 (선택, 예: 005930.KS)", value=""
).strip()
if custom_ticker:
    selected_ticker = custom_ticker
    selected_name = custom_ticker

period_label = st.sidebar.selectbox("기간", list(PERIOD_OPTIONS.keys()), index=3)
interval_label = st.sidebar.selectbox("봉 간격", list(INTERVAL_OPTIONS.keys()), index=0)

show_ma = st.sidebar.checkbox("이동평균선 표시 (20/60일)", value=True)
show_volume = st.sidebar.checkbox("거래량 표시", value=True)

st.sidebar.markdown("---")
compare_list = st.sidebar.multiselect(
    "여러 종목 비교 (수익률 %)",
    list(TICKERS.keys()),
    default=["Apple (AAPL)", "Microsoft (MSFT)", "NVIDIA (NVDA)"],
)

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance)")


# ---------------------------------------------------------
# 메인 타이틀
# ---------------------------------------------------------
st.title("🌍 글로벌 주요 주식 대시보드")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tab1, tab2, tab3 = st.tabs(["📌 시장 요약", "📈 종목 상세", "🔀 종목 비교"])


# ---------------------------------------------------------
# 탭 1: 시장 요약 (주요 종목 한눈에 보기)
# ---------------------------------------------------------
with tab1:
    st.subheader("주요 글로벌 종목 현황")
    with st.spinner("데이터를 불러오는 중..."):
        summary_df = load_multi_summary(TICKERS)

    if summary_df.empty:
        st.warning("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        cols = st.columns(4)
        for i, row in summary_df.iterrows():
            col = cols[i % 4]
            with col:
                st.metric(
                    label=row["종목명"],
                    value=f"{row['현재가']:,}",
                    delta=f"{row['등락률(%)']}%",
                )

        st.markdown("---")

        def _color_value(v):
            if isinstance(v, (int, float)):
                if v > 0:
                    return "color: red;"
                elif v < 0:
                    return "color: blue;"
            return ""

        try:
            styled_df = summary_df.style.map(_color_value, subset=["전일대비", "등락률(%)"])
        except AttributeError:
            # 구버전 pandas 호환 (map이 없는 경우 applymap 사용)
            styled_df = summary_df.style.applymap(_color_value, subset=["전일대비", "등락률(%)"])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------
# 탭 2: 종목 상세 (캔들차트 + 이동평균 + 거래량)
# ---------------------------------------------------------
with tab2:
    st.subheader(f"{selected_name} 상세 차트")

    period = PERIOD_OPTIONS[period_label]
    interval = INTERVAL_OPTIONS[interval_label]

    with st.spinner("차트 데이터를 불러오는 중..."):
        df = load_price_history(selected_ticker, period, interval)
        info = load_ticker_info(selected_ticker)

    if df is None or df.empty:
        st.error("해당 티커의 데이터를 찾을 수 없습니다. 티커명을 확인해주세요.")
    else:
        # 상단 지표
        last_price = df["Close"].iloc[-1]
        first_price = df["Close"].iloc[0]
        change = last_price - first_price
        pct_change = (change / first_price) * 100 if first_price else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가", f"{last_price:,.2f}", f"{pct_change:.2f}%")
        c2.metric("기간 내 최고가", f"{df['High'].max():,.2f}")
        c3.metric("기간 내 최저가", f"{df['Low'].min():,.2f}")
        c4.metric(
            "평균 거래량",
            f"{df['Volume'].mean():,.0f}" if "Volume" in df.columns else "N/A",
        )

        if info:
            with st.expander("기업/종목 정보"):
                info_cols = st.columns(3)
                info_cols[0].write(f"**섹터:** {info.get('sector', 'N/A')}")
                info_cols[1].write(f"**산업:** {info.get('industry', 'N/A')}")
                info_cols[2].write(f"**시가총액:** {info.get('marketCap', 'N/A'):,}"
                                    if isinstance(info.get('marketCap'), (int, float))
                                    else "**시가총액:** N/A")

        # 이동평균 계산
        if show_ma:
            df["MA20"] = df["Close"].rolling(window=20).mean()
            df["MA60"] = df["Close"].rolling(window=60).mean()

        # 서브플롯 구성 (캔들 + 거래량)
        row_heights = [0.75, 0.25] if show_volume else [1.0]
        rows = 2 if show_volume else 1

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=selected_name,
                increasing_line_color="red",
                decreasing_line_color="blue",
            ),
            row=1,
            col=1,
        )

        if show_ma:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["MA20"], name="MA20",
                    line=dict(color="orange", width=1.2),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["MA60"], name="MA60",
                    line=dict(color="purple", width=1.2),
                ),
                row=1, col=1,
            )

        if show_volume:
            colors = [
                "red" if row["Close"] >= row["Open"] else "blue"
                for _, row in df.iterrows()
            ]
            fig.add_trace(
                go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=colors),
                row=2, col=1,
            )

        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig.update_yaxes(title_text="가격", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="거래량", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("원본 데이터 보기"):
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------
# 탭 3: 종목 비교 (정규화 수익률 라인 차트)
# ---------------------------------------------------------
with tab3:
    st.subheader("종목별 수익률 비교 (기간 내 % 변화)")

    if not compare_list:
        st.info("사이드바에서 비교할 종목을 하나 이상 선택해주세요.")
    else:
        period = PERIOD_OPTIONS[period_label]
        interval = INTERVAL_OPTIONS[interval_label]

        fig_cmp = go.Figure()
        with st.spinner("비교 데이터를 불러오는 중..."):
            for name in compare_list:
                symbol = TICKERS[name]
                hist = load_price_history(symbol, period, interval)
                if hist is None or hist.empty:
                    continue
                normalized = (hist["Close"] / hist["Close"].iloc[0] - 1) * 100
                fig_cmp.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode="lines",
                        name=name,
                    )
                )

        fig_cmp.update_layout(
            height=600,
            template="plotly_white",
            yaxis_title="수익률 (%)",
            xaxis_title="날짜",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_cmp, use_container_width=True)


# ---------------------------------------------------------
# 푸터
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ 본 대시보드는 정보 제공 목적이며 투자 조언이 아닙니다. "
    "데이터는 Yahoo Finance(yfinance)를 통해 제공되며 지연될 수 있습니다."
)
