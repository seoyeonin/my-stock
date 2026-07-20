import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="지속가능경영 분석",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 지속가능경영 대표 종목 리스트
# ---------------------------------------------------------
ESG_TICKERS = {
    "Tesla (TSLA)": "TSLA",
    "NextEra Energy (NEE)": "NEE",
    "Microsoft (MSFT)": "MSFT",
    "Apple (AAPL)": "AAPL",
    "Ørsted (ORSTED.CO)": "ORSTED.CO",
    "Vestas Wind Systems (VWS.CO)": "VWS.CO",
    "Enphase Energy (ENPH)": "ENPH",
    "First Solar (FSLR)": "FSLR",
    "Waste Management (WM)": "WM",
    "Unilever (UL)": "UL",
    "Schneider Electric (SU.PA)": "SU.PA",
    "Iberdrola (IBE.MC)": "IBE.MC",
    "LG에너지솔루션 (373220.KS)": "373220.KS",
    "삼성SDI (006400.KS)": "006400.KS",
    "SK이노베이션 (096770.KS)": "096770.KS",
}

# 벤치마크 ETF
BENCHMARK_TICKERS = {
    "MSCI ESG 리더스 ETF (ESGU)": "ESGU",
    "청정에너지 ETF (ICLN)": "ICLN",
    "S&P 500 (SPY)": "SPY",
}

PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "5년": "5y",
    "연초 이후": "ytd",
}

INTERVAL_OPTIONS = {
    "1일": "1d",
    "1주": "1wk",
}

# ---------------------------------------------------------
# 데이터 로딩 함수
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

def _period_return(hist: pd.DataFrame, lookback: int) -> float:
    try:
        if len(hist) <= lookback:
            start_price = hist["Close"].iloc[0]
        else:
            start_price = hist["Close"].iloc[-lookback - 1]
        end_price = hist["Close"].iloc[-1]
        return round((end_price / start_price - 1) * 100, 2)
    except Exception:
        return np.nan

@st.cache_data(ttl=300)
def load_summary(ticker_map: dict) -> pd.DataFrame:
    rows = []
    for name, symbol in ticker_map.items():
        try:
            hist = yf.Ticker(symbol).history(period="6mo")
            if hist.empty or len(hist) < 2:
                continue

            info = yf.Ticker(symbol).info
            last_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2]
            pct_change = (last_close - prev_close) / prev_close * 100

            rows.append({
                "종목명": name,
                "티커": symbol,
                "현재가": round(last_close, 2),
                "등락률(%)": round(pct_change, 2),
                "1개월수익률(%)": _period_return(hist, 21),
                "3개월수익률(%)": _period_return(hist, 63),
                "6개월수익률(%)": _period_return(hist, len(hist)-1),
                "배당수익률(%)": round(info.get("dividendYield", 0) * 100, 2)
                    if isinstance(info.get("dividendYield"), (int, float)) else None,
                "PER": round(info.get("trailingPE"), 2)
                    if isinstance(info.get("trailingPE"), (int, float)) else None,
                "시가총액(B)": round(info.get("marketCap", 0) / 1e9, 1)
                    if isinstance(info.get("marketCap"), (int, float)) else None,
                "부채비율": round(info.get("debtToEquity"), 1)
                    if isinstance(info.get("debtToEquity"), (int, float)) else None,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
st.sidebar.title("🌱 지속가능경영 분석 설정")

selected_name = st.sidebar.selectbox(
    "분석 종목 선택", list(ESG_TICKERS.keys()), index=0
)
selected_ticker = ESG_TICKERS[selected_name]

period_label = st.sidebar.selectbox("기간", list(PERIOD_OPTIONS.keys()), index=3)
interval_label = st.sidebar.selectbox("봉 간격", list(INTERVAL_OPTIONS.keys()), index=0)

st.sidebar.markdown("---")

compare_list = st.sidebar.multiselect(
    "비교 종목 선택",
    list(ESG_TICKERS.keys()),
    default=["Tesla (TSLA)", "NextEra Energy (NEE)", "Microsoft (MSFT)"],
)

benchmark_name = st.sidebar.selectbox(
    "벤치마크 지수 선택",
    list(BENCHMARK_TICKERS.keys()),
    index=0,
)

include_benchmark = st.sidebar.checkbox("벤치마크 비교 포함", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance)")
st.sidebar.caption("⚠️ 투자 조언이 아닌 정보 제공 목적입니다.")

# ---------------------------------------------------------
# 메인 타이틀
# ---------------------------------------------------------
st.title("🌱 지속가능경영 주식 전문 분석")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 탭 구성 (ESG 탭 제거)
tab1, tab2, tab3 = st.tabs([
    "📌 섹터 전체 현황",
    "📈 종목 심층 분석",
    "🔀 종목 비교"
])

# ---------------------------------------------------------
# 탭 1: 섹터 전체 현황
# ---------------------------------------------------------
with tab1:
    st.subheader("지속가능경영 섹터 전체 현황")

    with st.spinner("섹터 데이터를 불러오는 중..."):
        summary_df = load_summary(ESG_TICKERS)

    if summary_df.empty:
        st.warning("데이터를 불러오지 못했습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "섹터 평균 1개월 수익률",
            f"{summary_df['1개월수익률(%)'].mean():.2f}%"
        )

        c2.metric(
            "섹터 평균 3개월 수익률",
            f"{summary_df['3개월수익률(%)'].mean():.2f}%"
        )

        best = summary_df.loc[summary_df['1개월수익률(%)'].idxmax()]
        c3.metric(
            "최고 성과 종목",
            best['종목명'],
            f"{best['1개월수익률(%)']:.2f}%"
        )

        avg_div = summary_df['배당수익률(%)'].mean(skipna=True)
        c4.metric(
            "평균 배당수익률",
            f"{avg_div:.2f}%" if not pd.isna(avg_div) else "N/A"
        )

        st.markdown("---")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("1개월 수익률 랭킹")

        rank_df = summary_df.sort_values("1개월수익률(%)", ascending=True)

        fig_rank = go.Figure(
            go.Bar(
                x=rank_df["1개월수익률(%)"],
                y=rank_df["종목명"],
                orientation="h",
                marker_color=[
                    "red" if v >= 0 else "blue"
                    for v in rank_df["1개월수익률(%)"]
                ]
            )
        )

        fig_rank.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="1개월 수익률 (%)"
        )

        st.plotly_chart(fig_rank, use_container_width=True)

# ---------------------------------------------------------
# 탭 2: 종목 심층 분석
# ---------------------------------------------------------
with tab2:
    st.subheader(f"{selected_name} 심층 분석")

    period = PERIOD_OPTIONS[period_label]
    interval = INTERVAL_OPTIONS[interval_label]

    with st.spinner("데이터를 불러오는 중..."):
        df = load_price_history(selected_ticker, period, interval)
        info = load_ticker_info(selected_ticker)

    if df is None or df.empty:
        st.error("데이터를 찾을 수 없습니다.")
    else:
        last_price = df["Close"].iloc[-1]
        first_price = df["Close"].iloc[0]
        pct_change = (last_price / first_price - 1) * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("현재가", f"{last_price:,.2f}", f"{pct_change:.2f}%")

        div_yield = info.get("dividendYield")
        c2.metric(
            "배당수익률",
            f"{div_yield*100:.2f}%" if isinstance(div_yield, (int, float)) else "N/A"
        )

        debt_eq = info.get("debtToEquity")
        c3.metric(
            "부채비율",
            f"{debt_eq:.1f}" if isinstance(debt_eq, (int, float)) else "N/A"
        )

        mcap = info.get("marketCap")
        c4.metric(
            "시가총액",
            f"${mcap/1e9:.1f}B" if isinstance(mcap, (int, float)) else "N/A"
        )

        with st.expander("기업 개요 및 재무 정보"):
            st.write(f"**섹터:** {info.get('sector', 'N/A')}")
            st.write(f"**산업:** {info.get('industry', 'N/A')}")
            st.write(f"**국가:** {info.get('country', 'N/A')}")
            st.write(f"**PER:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**ROE:** {info.get('returnOnEquity', 'N/A')}")

        # 이동평균
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()

        # 캔들차트
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25]
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
                decreasing_line_color="blue"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                name="MA20",
                line=dict(color="orange")
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA60"],
                name="MA60",
                line=dict(color="purple")
            ),
            row=1, col=1
        )

        vol_colors = [
            "red" if row["Close"] >= row["Open"] else "blue"
            for _, row in df.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=vol_colors,
                name="거래량"
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=700,
            template="plotly_white",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # RSI
        st.subheader("RSI (14일)")

        df["RSI"] = compute_rsi(df["Close"])

        fig_rsi = go.Figure()

        fig_rsi.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI",
                line=dict(color="teal")
            )
        )

        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")

        fig_rsi.update_layout(
            height=300,
            template="plotly_white",
            yaxis_title="RSI"
        )

        st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------------------------------------------
# 탭 3: 종목 비교
# ---------------------------------------------------------
with tab3:
    st.subheader("종목 수익률 비교")

    if not compare_list:
        st.info("비교할 종목을 선택해주세요.")
    else:
        period = PERIOD_OPTIONS[period_label]
        interval = INTERVAL_OPTIONS[interval_label]

        fig_cmp = go.Figure()

        for name in compare_list:
            symbol = ESG_TICKERS[name]
            hist = load_price_history(symbol, period, interval)

            if hist is not None and not hist.empty:
                normalized = (hist["Close"] / hist["Close"].iloc[0] - 1) * 100

                fig_cmp.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode="lines",
                        name=name
                    )
                )

        # 벤치마크 추가
        if include_benchmark:
            bench = BENCHMARK_TICKERS[benchmark_name]
            bench_hist = load_price_history(bench, period, interval)

            if bench_hist is not None and not bench_hist.empty:
                bench_norm = (
                    bench_hist["Close"] / bench_hist["Close"].iloc[0] - 1
                ) * 100

                fig_cmp.add_trace(
                    go.Scatter(
                        x=bench_hist.index,
                        y=bench_norm,
                        mode="lines",
                        name=benchmark_name,
                        line=dict(color="black", width=2, dash="dash")
                    )
                )

        fig_cmp.update_layout(
            height=600,
            template="plotly_white",
            yaxis_title="수익률 (%)",
            xaxis_title="날짜"
        )

        fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_cmp, use_container_width=True)

        # 상관관계
        st.subheader("종목간 수익률 상관관계")

        price_data = {}

        for name in compare_list:
            symbol = ESG_TICKERS[name]
            hist = load_price_history(symbol, period, interval)

            if hist is not None and not hist.empty:
                price_data[name] = hist["Close"].pct_change()

        if len(price_data) >= 2:
            corr_df = pd.DataFrame(price_data).corr()

            fig_heat = go.Figure(
                go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.columns,
                    colorscale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_df.values, 2),
                    texttemplate="%{text}"
                )
            )

            fig_heat.update_layout(height=500)
            st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------
# 푸터
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ 본 페이지는 정보 제공 목적이며 투자 조언이 아닙니다. "
    "데이터는 Yahoo Finance(yfinance)를 통해 제공됩니다."
)
