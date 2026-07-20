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
    page_title="금융/자산운용 분석",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 금융/자산운용 대표 종목 리스트
# 대형은행, 투자은행, 자산운용사, 보험, 카드사, 국내 금융지주 등
# ---------------------------------------------------------
FINANCE_TICKERS = {
    "JPMorgan Chase (JPM)": "JPM",
    "Bank of America (BAC)": "BAC",
    "Wells Fargo (WFC)": "WFC",
    "Goldman Sachs (GS)": "GS",
    "Morgan Stanley (MS)": "MS",
    "Citigroup (C)": "C",
    "BlackRock (BLK)": "BLK",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Visa (V)": "V",
    "Mastercard (MA)": "MA",
    "American Express (AXP)": "AXP",
    "KB금융 (105560.KS)": "105560.KS",
    "신한지주 (055550.KS)": "055550.KS",
    "하나금융지주 (086790.KS)": "086790.KS",
    "미래에셋증권 (006800.KS)": "006800.KS",
}

# 섹터 벤치마크
BENCHMARK_TICKERS = {
    "금융섹터 ETF (XLF)": "XLF",
    "지역은행 ETF (KRE)": "KRE",
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
            change = last_close - prev_close
            pct_change = (change / prev_close) * 100

            ret_1m = _period_return(hist, 21)
            ret_3m = _period_return(hist, 63)
            ret_6m = _period_return(hist, len(hist) - 1)

            volatility = hist["Close"].pct_change().std() * np.sqrt(252) * 100

            rows.append(
                {
                    "종목명": name,
                    "티커": symbol,
                    "현재가": round(last_close, 2),
                    "등락률(%)": round(pct_change, 2),
                    "1개월수익률(%)": ret_1m,
                    "3개월수익률(%)": ret_3m,
                    "6개월수익률(%)": ret_6m,
                    "PER": round(info.get("trailingPE"), 2) if isinstance(info.get("trailingPE"), (int, float)) else None,
                    "PBR": round(info.get("priceToBook"), 2) if isinstance(info.get("priceToBook"), (int, float)) else None,
                    "배당수익률(%)": round(info.get("dividendYield", 0) * 100, 2) if isinstance(info.get("dividendYield"), (int, float)) else None,
                    "ROE(%)": round(info.get("returnOnEquity", 0) * 100, 2) if isinstance(info.get("returnOnEquity"), (int, float)) else None,
                    "시가총액(B)": round(info.get("marketCap", 0) / 1e9, 1) if isinstance(info.get("marketCap"), (int, float)) else None,
                    "변동성(연간,%)": round(volatility, 1),
                }
            )
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
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
st.sidebar.title("🏦 금융/자산운용 분석 설정")

selected_name = st.sidebar.selectbox(
    "분석 종목 선택", list(FINANCE_TICKERS.keys()), index=0
)
selected_ticker = FINANCE_TICKERS[selected_name]

period_label = st.sidebar.selectbox("기간", list(PERIOD_OPTIONS.keys()), index=3)
interval_label = st.sidebar.selectbox("봉 간격", list(INTERVAL_OPTIONS.keys()), index=0)

st.sidebar.markdown("---")
compare_list = st.sidebar.multiselect(
    "비교 종목 선택",
    list(FINANCE_TICKERS.keys()),
    default=["JPMorgan Chase (JPM)", "Goldman Sachs (GS)", "Visa (V)", "KB금융 (105560.KS)"],
)

benchmark_name = st.sidebar.selectbox(
    "벤치마크 지수 선택", list(BENCHMARK_TICKERS.keys()), index=0
)
include_benchmark = st.sidebar.checkbox("벤치마크 비교 포함", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance)")
st.sidebar.caption("⚠️ 투자 조언이 아닌 정보 제공 목적입니다.")


# ---------------------------------------------------------
# 메인 타이틀
# ---------------------------------------------------------
st.title("🏦 금융/자산운용 주식 전문 분석")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(
    "글로벌 대형은행, 투자은행, 자산운용사, 카드사, 국내 금융지주 등 "
    "금융 섹터 핵심 종목을 밸류에이션, 수익성, 섹터 벤치마크 관점에서 분석합니다."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 섹터 전체 현황", "💰 밸류에이션 분석", "📈 종목 심층 분석", "🔀 종목 비교"]
)


# ---------------------------------------------------------
# 탭 1: 섹터 전체 현황
# ---------------------------------------------------------
with tab1:
    st.subheader("금융/자산운용 섹터 전체 현황")
    with st.spinner("섹터 데이터를 불러오는 중..."):
        summary_df = load_summary(FINANCE_TICKERS)

    if summary_df.empty:
        st.warning("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        avg_1m = summary_df["1개월수익률(%)"].mean()
        avg_3m = summary_df["3개월수익률(%)"].mean()
        best_performer = summary_df.loc[summary_df["1개월수익률(%)"].idxmax()]
        avg_div = summary_df["배당수익률(%)"].mean(skipna=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("섹터 평균 1개월 수익률", f"{avg_1m:.2f}%")
        c2.metric("섹터 평균 3개월 수익률", f"{avg_3m:.2f}%")
        c3.metric("최고 성과 (1개월)", best_performer["종목명"], f"{best_performer['1개월수익률(%)']:.2f}%")
        c4.metric("섹터 평균 배당수익률", f"{avg_div:.2f}%" if not pd.isna(avg_div) else "N/A")

        st.markdown("---")

        def _color_value(v):
            if isinstance(v, (int, float)):
                if v > 0:
                    return "color: red;"
                elif v < 0:
                    return "color: blue;"
            return ""

        pct_cols = ["등락률(%)", "1개월수익률(%)", "3개월수익률(%)", "6개월수익률(%)"]
        try:
            styled_df = summary_df.style.map(_color_value, subset=pct_cols)
        except AttributeError:
            styled_df = summary_df.style.applymap(_color_value, subset=pct_cols)

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("1개월 수익률 랭킹")
        rank_df = summary_df.sort_values("1개월수익률(%)", ascending=True)
        fig_rank = go.Figure(
            go.Bar(
                x=rank_df["1개월수익률(%)"],
                y=rank_df["종목명"],
                orientation="h",
                marker_color=["red" if v >= 0 else "blue" for v in rank_df["1개월수익률(%)"]],
            )
        )
        fig_rank.update_layout(
            height=500, template="plotly_white",
            xaxis_title="1개월 수익률 (%)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_rank, use_container_width=True)


# ---------------------------------------------------------
# 탭 2: 밸류에이션 분석 (PER, PBR, ROE, 배당)
# ---------------------------------------------------------
with tab2:
    st.subheader("금융주 밸류에이션 비교")
    st.caption(
        "금융주는 자산 대비 가치평가가 중요하므로 PER 외에 PBR(주가순자산비율)과 ROE(자기자본이익률)를 함께 봅니다. "
        "일반적으로 PBR이 낮고 ROE가 높을수록 저평가된 우량 금융주로 해석됩니다."
    )

    if summary_df.empty:
        st.warning("데이터를 불러오지 못했습니다.")
    else:
        valuation_df = summary_df.dropna(subset=["PER", "PBR"])
        if not valuation_df.empty:
            fig_val = go.Figure(
                go.Scatter(
                    x=valuation_df["PBR"],
                    y=valuation_df["ROE(%)"],
                    mode="markers+text",
                    text=valuation_df["종목명"],
                    textposition="top center",
                    marker=dict(
                        size=valuation_df["시가총액(B)"].fillna(10).clip(5, 60),
                        color=valuation_df["배당수익률(%)"].fillna(0),
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="배당<br>수익률(%)"),
                        sizemode="area",
                    ),
                )
            )
            fig_val.update_layout(
                height=550, template="plotly_white",
                xaxis_title="PBR (주가순자산비율, 배)",
                yaxis_title="ROE (자기자본이익률, %)",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_val, use_container_width=True)
            st.caption("ℹ️ 버블 크기는 시가총액을 나타냅니다.")
        else:
            st.info("PBR/ROE 데이터가 부족하여 산점도를 표시할 수 없습니다.")

        st.markdown("---")
        st.subheader("PER 비교")
        per_df = summary_df.dropna(subset=["PER"]).sort_values("PER", ascending=True)
        if not per_df.empty:
            fig_per = go.Figure(
                go.Bar(
                    x=per_df["PER"],
                    y=per_df["종목명"],
                    orientation="h",
                    marker_color="steelblue",
                )
            )
            fig_per.update_layout(
                height=500, template="plotly_white",
                xaxis_title="PER (배)",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_per, use_container_width=True)

        st.markdown("---")
        st.subheader("배당수익률 비교")
        div_df = summary_df.dropna(subset=["배당수익률(%)"]).sort_values("배당수익률(%)", ascending=True)
        if not div_df.empty:
            fig_div = go.Figure(
                go.Bar(
                    x=div_df["배당수익률(%)"],
                    y=div_df["종목명"],
                    orientation="h",
                    marker_color="darkgreen",
                )
            )
            fig_div.update_layout(
                height=500, template="plotly_white",
                xaxis_title="배당수익률 (%)",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_div, use_container_width=True)


# ---------------------------------------------------------
# 탭 3: 종목 심층 분석
# ---------------------------------------------------------
with tab3:
    st.subheader(f"{selected_name} 심층 분석")

    period = PERIOD_OPTIONS[period_label]
    interval = INTERVAL_OPTIONS[interval_label]

    with st.spinner("데이터를 불러오는 중..."):
        df = load_price_history(selected_ticker, period, interval)
        info = load_ticker_info(selected_ticker)

    if df is None or df.empty:
        st.error("해당 티커의 데이터를 찾을 수 없습니다.")
    else:
        last_price = df["Close"].iloc[-1]
        first_price = df["Close"].iloc[0]
        pct_change = (last_price / first_price - 1) * 100 if first_price else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("현재가", f"{last_price:,.2f}", f"{pct_change:.2f}%")
        pe = info.get("trailingPE")
        c2.metric("PER", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
        pb = info.get("priceToBook")
        c3.metric("PBR", f"{pb:.2f}" if isinstance(pb, (int, float)) else "N/A")
        roe = info.get("returnOnEquity")
        c4.metric("ROE", f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A")
        div_yield = info.get("dividendYield")
        c5.metric("배당수익률", f"{div_yield*100:.2f}%" if isinstance(div_yield, (int, float)) else "N/A")

        with st.expander("기업 개요 및 핵심 재무지표"):
            info_c1, info_c2, info_c3 = st.columns(3)
            info_c1.write(f"**섹터:** {info.get('sector', 'N/A')}")
            info_c1.write(f"**산업:** {info.get('industry', 'N/A')}")
            info_c1.write(f"**본사:** {info.get('country', 'N/A')}")

            mcap = info.get("marketCap")
            info_c2.write(f"**시가총액:** ${mcap/1e9:.1f}B" if isinstance(mcap, (int, float)) else "**시가총액:** N/A")
            info_c2.write(f"**52주 최고가:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
            info_c2.write(f"**52주 최저가:** {info.get('fiftyTwoWeekLow', 'N/A')}")

            info_c3.write(f"**주당순이익(EPS):** {info.get('trailingEps', 'N/A')}")
            info_c3.write(f"**베타(변동성):** {info.get('beta', 'N/A')}")
            info_c3.write(f"**총 부채:** {info.get('totalDebt', 'N/A')}")

        # 캔들차트 + 이동평균 + 거래량
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA60"] = df["Close"].rollin
