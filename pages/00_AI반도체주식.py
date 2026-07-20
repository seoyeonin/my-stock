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
    page_title="AI 반도체 분석",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# AI 반도체 관련 종목 리스트
# ---------------------------------------------------------
AI_CHIP_TICKERS = {
    "NVIDIA (NVDA)": "NVDA",
    "AMD (AMD)": "AMD",
    "Intel (INTC)": "INTC",
    "TSMC (TSM)": "TSM",
    "Broadcom (AVGO)": "AVGO",
    "Qualcomm (QCOM)": "QCOM",
    "Micron (MU)": "MU",
    "Samsung Electronics (005930.KS)": "005930.KS",
    "SK Hynix (000660.KS)": "000660.KS",
    "ASML (ASML)": "ASML",
    "Arm Holdings (ARM)": "ARM",
    "Marvell Technology (MRVL)": "MRVL",
    "Applied Materials (AMAT)": "AMAT",
    "Lam Research (LRCX)": "LRCX",
    "KLA Corporation (KLAC)": "KLAC",
}

# 참고 지수 (반도체 섹터 벤치마크)
BENCHMARK_TICKERS = {
    "필라델피아 반도체지수 (SOXX)": "SOXX",
    "나스닥100 (QQQ)": "QQQ",
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

            high_52w = hist["High"].max()
            low_52w = hist["Low"].min()

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
                    "시가총액(B)": round(info.get("marketCap", 0) / 1e9, 1) if isinstance(info.get("marketCap"), (int, float)) else None,
                    "변동성(연간,%)": round(volatility, 1),
                    "52주최고": round(high_52w, 2),
                    "52주최저": round(low_52w, 2),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


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
st.sidebar.title("🧠 AI 반도체 분석 설정")

selected_name = st.sidebar.selectbox(
    "분석 종목 선택", list(AI_CHIP_TICKERS.keys()), index=0
)
selected_ticker = AI_CHIP_TICKERS[selected_name]

period_label = st.sidebar.selectbox("기간", list(PERIOD_OPTIONS.keys()), index=3)
interval_label = st.sidebar.selectbox("봉 간격", list(INTERVAL_OPTIONS.keys()), index=0)

st.sidebar.markdown("---")
compare_list = st.sidebar.multiselect(
    "비교 종목 선택",
    list(AI_CHIP_TICKERS.keys()),
    default=["NVIDIA (NVDA)", "AMD (AMD)", "TSMC (TSM)", "Samsung Electronics (005930.KS)"],
)

include_benchmark = st.sidebar.checkbox("필라델피아 반도체지수(SOXX) 비교 포함", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance)")
st.sidebar.caption("⚠️ 투자 조언이 아닌 정보 제공 목적입니다.")


# ---------------------------------------------------------
# 메인 타이틀
# ---------------------------------------------------------
st.title("🧠 AI 반도체 주식 전문 분석")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(
    "NVIDIA, AMD, TSMC, 삼성전자, SK하이닉스 등 AI 반도체 밸류체인 핵심 종목을 "
    "재무지표, 기술적 지표, 섹터 비교 관점에서 분석합니다."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 섹터 전체 현황", "📈 종목 심층 분석", "🔀 종목 비교", "🧪 기술적 지표"]
)


# ---------------------------------------------------------
# 탭 1: 섹터 전체 현황
# ---------------------------------------------------------
with tab1:
    st.subheader("AI 반도체 섹터 전체 현황")
    with st.spinner("섹터 데이터를 불러오는 중..."):
        summary_df = load_summary(AI_CHIP_TICKERS)

    if summary_df.empty:
        st.warning("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        avg_1m = summary_df["1개월수익률(%)"].mean()
        avg_3m = summary_df["3개월수익률(%)"].mean()
        best_performer = summary_df.loc[summary_df["1개월수익률(%)"].idxmax()]
        worst_performer = summary_df.loc[summary_df["1개월수익률(%)"].idxmin()]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("섹터 평균 1개월 수익률", f"{avg_1m:.2f}%")
        c2.metric("섹터 평균 3개월 수익률", f"{avg_3m:.2f}%")
        c3.metric("최고 성과 (1개월)", best_performer["종목명"], f"{best_performer['1개월수익률(%)']:.2f}%")
        c4.metric("최저 성과 (1개월)", worst_performer["종목명"], f"{worst_performer['1개월수익률(%)']:.2f}%")

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
                marker_color=[
                    "red" if v >= 0 else "blue" for v in rank_df["1개월수익률(%)"]
                ],
            )
        )
        fig_rank.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="1개월 수익률 (%)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.subheader("시가총액 vs PER 분포")
        valid_df = summary_df.dropna(subset=["PER", "시가총액(B)"])
        if not valid_df.empty:
            fig_scatter = go.Figure(
                go.Scatter(
                    x=valid_df["시가총액(B)"],
                    y=valid_df["PER"],
                    mode="markers+text",
                    text=valid_df["종목명"],
                    textposition="top center",
                    marker=dict(size=14, color=valid_df["1개월수익률(%)"], colorscale="RdBu_r", showscale=True, colorbar=dict(title="1개월<br>수익률(%)")),
                )
            )
            fig_scatter.update_layout(
                height=550,
                template="plotly_white",
                xaxis_title="시가총액 (10억 달러)",
                yaxis_title="PER (배)",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("PER 또는 시가총액 데이터가 부족하여 산점도를 표시할 수 없습니다.")


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
        st.error("해당 티커의 데이터를 찾을 수 없습니다.")
    else:
        last_price = df["Close"].iloc[-1]
        first_price = df["Close"].iloc[0]
        pct_change = (last_price / first_price - 1) * 100 if first_price else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("현재가", f"{last_price:,.2f}", f"{pct_change:.2f}%")
        c2.metric("기간 최고가", f"{df['High'].max():,.2f}")
        c3.metric("기간 최저가", f"{df['Low'].min():,.2f}")
        pe = info.get("trailingPE")
        c4.metric("PER", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
        mcap = info.get("marketCap")
        c5.metric("시가총액", f"${mcap/1e9:.1f}B" if isinstance(mcap, (int, float)) else "N/A")

        with st.expander("기업 개요 및 핵심 재무지표"):
            info_c1, info_c2, info_c3 = st.columns(3)
            info_c1.write(f"**섹터:** {info.get('sector', 'N/A')}")
            info_c1.write(f"**산업:** {info.get('industry', 'N/A')}")
            info_c1.write(f"**본사:** {info.get('country', 'N/A')}")

            info_c2.write(f"**52주 최고가:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
            info_c2.write(f"**52주 최저가:** {info.get('fiftyTwoWeekLow', 'N/A')}")
            info_c2.write(f"**배당수익률:** {info.get('dividendYield', 'N/A')}")

            info_c3.write(f"**매출성장률(YoY):** {info.get('revenueGrowth', 'N/A')}")
            info_c3.write(f"**영업이익률:** {info.get('operatingMargins', 'N/A')}")
            info_c3.write(f"**ROE:** {info.get('returnOnEquity', 'N/A')}")

        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA60"] = df["Close"].rolling(window=60).mean()

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=selected_name,
                increasing_line_color="red", decreasing_line_color="blue",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="orange", width=1.2)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MA60"], name="MA60", line=dict(color="purple", width=1.2)),
            row=1, col=1,
        )
        vol_colors = ["red" if row["Close"] >= row["Open"] else "blue" for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=vol_colors),
            row=2, col=1,
        )
        fig.update_layout(
            height=700, xaxis_rangeslider_visible=False, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 탭 3: 종목 비교
# ---------------------------------------------------------
with tab3:
    st.subheader("AI 반도체 종목 수익률 비교")

    if not compare_list:
        st.info("사이드바에서 비교할 종목을 하나 이상 선택해주세요.")
    else:
        period = PERIOD_OPTIONS[period_label]
        interval = INTERVAL_OPTIONS[interval_label]

        fig_cmp = go.Figure()
        with st.spinner("비교 데이터를 불러오는 중..."):
            for name in compare_list:
                symbol = AI_CHIP_TICKERS[name]
                hist = load_price_history(symbol, period, interval)
                if hist is None or hist.empty:
                    continue
                normalized = (hist["Close"] / hist["Close"].iloc[0] - 1) * 100
                fig_cmp.add_trace(go.Scatter(x=hist.index, y=normalized, mode="lines", name=name))

            if include_benchmark:
                bench_hist = load_price_history("SOXX", period, interval)
                if bench_hist is not None and not bench_hist.empty:
                    bench_norm = (bench_hist["Close"] / bench_hist["Close"].iloc[0] - 1) * 100
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=bench_hist.index, y=bench_norm, mode="lines",
                            name="필라델피아 반도체지수 (SOXX)",
                            line=dict(color="black", width=2, dash="dash"),
                        )
                    )

        fig_cmp.update_layout(
            height=600, template="plotly_white",
            yaxis_title="수익률 (%)", xaxis_title="날짜",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("종목간 수익률 상관관계")
        price_data = {}
        for name in compare_list:
            symbol = AI_CHIP_TICKERS[name]
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
                    zmin=-1, zmax=1,
                    text=np.round(corr_df.values, 2),
                    texttemplate="%{text}",
                )
            )
            fig_heat.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("상관관계 분석을 위해 2개 이상의 종목을 선택해주세요.")


# ---------------------------------------------------------
# 탭 4: 기술적 지표 (RSI, MACD)
# ---------------------------------------------------------
with tab4:
    st.subheader(f"{selected_name} 기술적 지표 (RSI / MACD)")

    period = PERIOD_OPTIONS[period_label]
    interval = INTERVAL_OPTIONS[interval_label]

    with st.spinner("기술적 지표 계산 중..."):
        df_tech = load_price_history(selected_ticker, period, interval)

    if df_tech is None or df_tech.empty:
        st.error("데이터를 불러올 수 없습니다.")
    else:
        df_tech["RSI"] = compute_rsi(df_tech["Close"])
        macd_line, signal_line, histogram = compute_macd(df_tech["Close"])
        df_tech["MACD"] = macd_line
        df_tech["Signal"] = signal_line
        df_tech["Histogram"] = histogram

        latest_rsi = df_tech["RSI"].iloc[-1]
        rsi_status = "과매수" if latest_rsi > 70 else ("과매도" if latest_rsi < 30 else "중립")

        c1, c2 = st.columns(2)
        c1.metric("현재 RSI (14일)", f"{latest_rsi:.1f}", rsi_status)
        macd_signal = "골든크로스(상승 신호)" if macd_line.iloc[-1] > signal_line.iloc[-1] else "데드크로스(하락 신호)"
        c2.metric("현재 MACD 신호", macd_signal)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_tech.index, y=df_tech["RSI"], name="RSI", line=dict(color="teal")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수(70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="과매도(30)")
        fig_rsi.update_layout(
            height=350, template="plotly_white", yaxis_title="RSI",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        fig_macd = make_subplots(specs=[[{"secondary_y": False}]])
        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech["MACD"], name="MACD", line=dict(color="blue")))
        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech["Signal"], name="Signal", line=dict(color="orange")))
        hist_colors = ["red" if v >= 0 else "blue" for v in df_tech["Histogram"]]
        fig_macd.add_trace(go.Bar(x=df_tech.index, y=df_tech["Histogram"], name="Histogram", marker_color=hist_colors))
        fig_macd.update_layout(
            height=350, template="plotly_white", yaxis_title="MACD",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_macd, use_container_width=True)

        st.caption(
            "ℹ️ RSI 70 이상은 과매수, 30 이하는 과매도 구간으로 해석됩니다. "
            "MACD가 시그널선을 상향 돌파하면 매수 신호, 하향 돌파하면 매도 신호로 흔히 해석되지만 "
            "단독 지표만으로 투자 판단을 내리는 것은 권장되지 않습니다."
        )


# ---------------------------------------------------------
# 푸터
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ 본 페이지는 정보 제공 목적이며 투자 조언이 아닙니다. "
    "데이터는 Yahoo Finance(yfinance)를 통해 제공되며 지연될 수 있습니다."
)
