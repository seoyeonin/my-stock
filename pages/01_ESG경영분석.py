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
    page_title="지속가능경영(ESG) 분석",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 지속가능경영(ESG) 대표 종목 리스트
# 재생에너지, ESG 우수기업, 전기차, 지속가능 소비재 등
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

# ESG 관련 ETF / 벤치마크
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


@st.cache_data(ttl=300)
def load_sustainability(ticker: str) -> pd.DataFrame:
    """yfinance의 ESG 점수 데이터 (제공되지 않는 종목도 있음)"""
    try:
        data = yf.Ticker(ticker).sustainability
        return data
    except Exception:
        return None


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

            esg_score = None
            try:
                sus = yf.Ticker(symbol).sustainability
                if sus is not None and not sus.empty and "totalEsg" in sus.index:
                    esg_score = sus.loc["totalEsg"].values[0]
            except Exception:
                pass

            rows.append(
                {
                    "종목명": name,
                    "티커": symbol,
                    "현재가": round(last_close, 2),
                    "등락률(%)": round(pct_change, 2),
                    "1개월수익률(%)": ret_1m,
                    "3개월수익률(%)": ret_3m,
                    "6개월수익률(%)": ret_6m,
                    "ESG위험점수": round(esg_score, 1) if isinstance(esg_score, (int, float)) else None,
                    "배당수익률(%)": round(info.get("dividendYield", 0) * 100, 2) if isinstance(info.get("dividendYield"), (int, float)) else None,
                    "PER": round(info.get("trailingPE"), 2) if isinstance(info.get("trailingPE"), (int, float)) else None,
                    "시가총액(B)": round(info.get("marketCap", 0) / 1e9, 1) if isinstance(info.get("marketCap"), (int, float)) else None,
                    "부채비율": round(info.get("debtToEquity"), 1) if isinstance(info.get("debtToEquity"), (int, float)) else None,
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


# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
st.sidebar.title("🌱 ESG 분석 설정")

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
    default=["Tesla (TSLA)", "NextEra Energy (NEE)", "Microsoft (MSFT)", "First Solar (FSLR)"],
)

benchmark_name = st.sidebar.selectbox(
    "벤치마크 지수 선택", list(BENCHMARK_TICKERS.keys()), index=0
)
include_benchmark = st.sidebar.checkbox("벤치마크 비교 포함", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance)")
st.sidebar.caption("⚠️ 투자 조언이 아닌 정보 제공 목적입니다.")
st.sidebar.caption(
    "ℹ️ ESG 위험점수는 Sustainalytics 기준이며, 낮을수록 ESG 리스크가 낮음을 의미합니다. "
    "일부 종목은 데이터가 제공되지 않을 수 있습니다."
)


# ---------------------------------------------------------
# 메인 타이틀
# ---------------------------------------------------------
st.title("🌱 지속가능경영(ESG) 주식 전문 분석")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(
    "재생에너지, 전기차, ESG 우수기업 등 지속가능경영 관련 핵심 종목을 "
    "ESG 위험점수, 재무 건전성, 섹터 벤치마크 관점에서 분석합니다."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 섹터 전체 현황", "🌍 ESG 스코어 분석", "📈 종목 심층 분석", "🔀 종목 비교"]
)


# ---------------------------------------------------------
# 탭 1: 섹터 전체 현황
# ---------------------------------------------------------
with tab1:
    st.subheader("지속가능경영 섹터 전체 현황")
    with st.spinner("섹터 데이터를 불러오는 중... (ESG 점수 조회로 다소 시간이 걸릴 수 있습니다)"):
        summary_df = load_summary(ESG_TICKERS)

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
# 탭 2: ESG 스코어 분석
# ---------------------------------------------------------
with tab2:
    st.subheader("ESG 위험점수 비교")
    st.caption(
        "Sustainalytics 기준 ESG 위험점수(totalEsg)를 사용합니다. "
        "점수가 낮을수록 ESG 관련 리스크가 낮은 기업입니다. (0~10: 무시가능, 10~20: 낮음, 20~30: 보통, 30~40: 높음, 40+: 심각)"
    )

    with st.spinner("ESG 데이터를 불러오는 중..."):
        esg_df = summary_df.dropna(subset=["ESG위험점수"]) if not summary_df.empty else pd.DataFrame()

    if esg_df.empty:
        st.warning(
            "선택한 종목들에 대한 ESG 점수 데이터를 Yahoo Finance에서 제공하지 않습니다. "
            "일부 해외/소형 종목은 ESG 데이터가 없을 수 있습니다."
        )
    else:
        esg_sorted = esg_df.sort_values("ESG위험점수", ascending=True)
        fig_esg = go.Figure(
            go.Bar(
                x=esg_sorted["ESG위험점수"],
                y=esg_sorted["종목명"],
                orientation="h",
                marker_color=esg_sorted["ESG위험점수"],
                marker=dict(
                    color=esg_sorted["ESG위험점수"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="ESG<br>위험점수"),
                ),
            )
        )
        fig_esg.update_layout(
            height=500, template="plotly_white",
            xaxis_title="ESG 위험점수 (낮을수록 우수)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_esg, use_container_width=True)

        st.markdown("---")
        st.subheader("ESG 점수 vs 주가 수익률")
        fig_scatter = go.Figure(
            go.Scatter(
                x=esg_df["ESG위험점수"],
                y=esg_df["3개월수익률(%)"],
                mode="markers+text",
                text=esg_df["종목명"],
                textposition="top center",
                marker=dict(
                    size=16,
                    color=esg_df["3개월수익률(%)"],
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="3개월<br>수익률(%)"),
                ),
            )
        )
        fig_scatter.update_layout(
            height=550, template="plotly_white",
            xaxis_title="ESG 위험점수 (낮을수록 우수)",
            yaxis_title="3개월 수익률 (%)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            "ℹ️ ESG 위험점수가 낮다고 해서 반드시 주가 수익률이 높은 것은 아닙니다. "
            "이 차트는 상관관계 참고용이며 인과관계를 의미하지 않습니다."
        )


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
        sus = load_sustainability(selected_ticker)

    if df is None or df.empty:
        st.error("해당 티커의 데이터를 찾을 수 없습니다.")
    else:
        last_price = df["Close"].iloc[-1]
        first_price = df["Close"].iloc[0]
        pct_change = (last_price / first_price - 1) * 100 if first_price else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가", f"{last_price:,.2f}", f"{pct_change:.2f}%")
        div_yield = info.get("dividendYield")
        c2.metric("배당수익률", f"{div_yield*100:.2f}%" if isinstance(div_yield, (int, float)) else "N/A")
        debt_eq = info.get("debtToEquity")
        c3.metric("부채비율", f"{debt_eq:.1f}" if isinstance(debt_eq, (int, float)) else "N/A")
        mcap = info.get("marketCap")
        c4.metric("시가총액", f"${mcap/1e9:.1f}B" if isinstance(mcap, (int, float)) else "N/A")

        with st.expander("기업 개요 및 재무 건전성"):
            info_c1, info_c2, info_c3 = st.columns(3)
            info_c1.write(f"**섹터:** {info.get('sector', 'N/A')}")
            info_c1.write(f"**산업:** {info.get('industry', 'N/A')}")
            info_c1.write(f"**본사:** {info.get('country', 'N/A')}")

            info_c2.write(f"**영업이익률:** {info.get('operatingMargins', 'N/A')}")
            info_c2.write(f"**ROE:** {info.get('returnOnEquity', 'N/A')}")
            info_c2.write(f"**유동비율:** {info.get('currentRatio', 'N/A')}")

            info_c3.write(f"**매출성장률(YoY):** {info.get('revenueGrowth', 'N/A')}")
            info_c3.write(f"**잉여현금흐름:** {info.get('freeCashflow', 'N/A')}")
            info_c3.write(f"**PER:** {info.get('trailingPE', 'N/A')}")

        if sus is not None and not sus.empty:
            with st.expander("🌱 ESG 세부 점수"):
                try:
                    display_rows = []
                    label_map = {
                        "totalEsg": "종합 ESG 위험점수",
                        "environmentScore": "환경(E) 점수",
                        "socialScore": "사회(S) 점수",
                        "governanceScore": "지배구조(G) 점수",
                        "esgPerformance": "ESG 성과 등급",
                        "peerGroup": "동종업계 그룹",
                    }
                    for key, label in label_map.items():
                        if key in sus.index:
                            display_rows.append({"항목": label, "값": sus.loc[key].values[0]})
                    if display_rows:
                        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("세부 ESG 항목 데이터가 제공되지 않습니다.")
                except Exception:
                    st.info("ESG 세부 데이터를 표시할 수 없습니다.")
        else:
            st.info("이 종목은 Yahoo Finance에서 ESG 세부 데이터를 제공하지 않습니다.")

        # 캔들차트 + 이동평균
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

        # RSI
        st.subheader("RSI (14일)")
        df["RSI"] = compute_rsi(df["Close"])
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="teal")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수(70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="과매도(30)")
        fig_rsi.update_layout(
            height=300, template="plotly_white", yaxis_title="RSI",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)


# ---------------------------------------------------------
# 탭 4: 종목 비교
# ---------------------------------------------------------
with tab4:
    st.subheader("ESG 종목 수익률 비교")

    if not compare_list:
        st.info("사이드바에서 비교할 종목을 하나 이상 선택해주세요.")
    else:
        period = PERIOD_OPTIONS[period_label]
        interval = INTERVAL_OPTIONS[interval_label]
        benchmark_ticker = BENCHMARK_TICKERS[benchmark_name]

        fig_cmp = go.Figure()
        with st.spinner("비교 데이터를 불러오는 중..."):
            for name in compare_list:
                symbol = ESG_TICKERS[name]
                hist = load_price_history(symbol, period, interval)
                if hist is None or hist.empty:
                    continue
                normalized = (hist["Close"] / hist["Close"].iloc[0] - 1) * 100
                fig_cmp.add_trace(go.Scatter(x=hist.index, y=normalized, mode="lines", name=name))

            if include_benchmark:
                bench_hist = load_price_history(benchmark_ticker, period, interval)
                if bench_hist is not None and not bench_hist.empty:
                    bench_norm = (bench_hist["Close"] / bench_hist["Close"].iloc[0] - 1) * 100
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=bench_hist.index, y=bench_norm, mode="lines",
                            name=benchmark_name,
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
# 푸터
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ 본 페이지는 정보 제공 목적이며 투자 조언이 아닙니다. "
    "데이터는 Yahoo Finance(yfinance)를 통해 제공되며 지연되거나 일부 종목은 제공되지 않을 수 있습니다."
)
