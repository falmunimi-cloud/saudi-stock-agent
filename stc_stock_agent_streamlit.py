import warnings
warnings.filterwarnings("ignore")

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands

# =========================
# CONFIG
# =========================
DEFAULT_MARKET_TICKER = "^TASI.SR"
EXPORT_DIR = "streamlit_outputs"
os.makedirs(EXPORT_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    "capital": 100000,
    "risk_per_trade": 0.01,
    "target_return": 0.01,
    "max_stop_pct": 0.006,
    "daily_period": "2y",
    "hourly_period": "730d",
    "m15_period": "60d",
    "daily_interval": "1d",
    "hourly_interval": "60m",
    "m15_interval": "15m",
    "relative_strength_window": 20,
    "rsi_floor": 48,
    "rsi_ceiling": 68,
    "min_score_buy": 68,
    "min_score_wait": 48,
    "backtest_hold_bars": 12,
    "sr_lookback": 20,
    "fake_breakout_window": 3,
}

# =========================
# HELPERS
# =========================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def ensure_ohlcv(df: pd.DataFrame):
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")


def fmt_num(x, nd=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{nd}f}"


def fmt_pct(x, nd=2):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:,.{nd}f}%"

# =========================
# DATA LOADER
# =========================
class MarketDataLoader:
    def __init__(self, stock_ticker: str, market_ticker: str):
        self.stock_ticker = stock_ticker
        self.market_ticker = market_ticker
        self.stock_obj = yf.Ticker(stock_ticker)
        self.market_obj = yf.Ticker(market_ticker)

    def download_prices(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        df = flatten_columns(df)
        df.dropna(inplace=True)
        ensure_ohlcv(df)
        return df

    def load_all(self, cfg: Dict) -> Dict[str, pd.DataFrame]:
        return {
            "stock_daily": self.download_prices(self.stock_ticker, cfg["daily_interval"], cfg["daily_period"]),
            "market_daily": self.download_prices(self.market_ticker, cfg["daily_interval"], cfg["daily_period"]),
            "stock_hourly": self.download_prices(self.stock_ticker, cfg["hourly_interval"], cfg["hourly_period"]),
            "market_hourly": self.download_prices(self.market_ticker, cfg["hourly_interval"], cfg["hourly_period"]),
            "stock_15m": self.download_prices(self.stock_ticker, cfg["m15_interval"], cfg["m15_period"]),
            "market_15m": self.download_prices(self.market_ticker, cfg["m15_interval"], cfg["m15_period"]),
        }

    def get_info_snapshot(self) -> Dict:
        info = self.stock_obj.info if isinstance(self.stock_obj.info, dict) else {}
        return {
            "symbol": self.stock_ticker,
            "shortName": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "currentPrice": info.get("currentPrice"),
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            "ebitdaMargins": info.get("ebitdaMargins"),
            "profitMargins": info.get("profitMargins"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "averageVolume": info.get("averageVolume"),
        }

# =========================
# FEATURE ENGINEERING
# =========================
class FeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None, rs_window: int = 20) -> pd.DataFrame:
        df = df.copy()
        ensure_ohlcv(df)

        df["sma_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["sma_100"] = SMAIndicator(df["Close"], window=100).sma_indicator()
        df["sma_200"] = SMAIndicator(df["Close"], window=200).sma_indicator()
        df["ema_12"] = EMAIndicator(df["Close"], window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(df["Close"], window=26).ema_indicator()

        df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
        macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
        df["atr_14"] = atr.average_true_range()
        df["atr_pct"] = df["atr_14"] / df["Close"]

        bb = BollingerBands(df["Close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

        adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["adx_14"] = adx.adx()

        df["ret_1"] = df["Close"].pct_change(1)
        df["ret_5"] = df["Close"].pct_change(5)
        df["ret_20"] = df["Close"].pct_change(20)
        df["vol_ma_20"] = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_ma_20"]

        spread = (df["High"] - df["Low"]).replace(0, np.nan)
        df["close_location"] = (df["Close"] - df["Low"]) / spread

        if market_df is not None:
            m = market_df[["Close"]].rename(columns={"Close": "m_close"}).copy()
            df = df.join(m, how="left")
            df["stock_ret_rs"] = df["Close"].pct_change(rs_window)
            df["market_ret_rs"] = df["m_close"].pct_change(rs_window)
            df["relative_strength"] = df["stock_ret_rs"] - df["market_ret_rs"]

        df.dropna(inplace=True)
        return df

# =========================
# PRICE ACTION ENGINE
# =========================
def compute_support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["resistance_20"] = out["High"].rolling(lookback).max().shift(1)
    out["support_20"] = out["Low"].rolling(lookback).min().shift(1)
    out["near_support"] = ((out["Close"] - out["support_20"]).abs() / out["Close"]) <= 0.004
    out["near_resistance"] = ((out["Close"] - out["resistance_20"]).abs() / out["Close"]) <= 0.004
    return out


def score_support_resistance_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    width = ((out["resistance_20"] - out["support_20"]) / out["Close"]).replace([np.inf, -np.inf], np.nan)
    out["support_quality"] = 0.0
    out["resistance_quality"] = 0.0
    out.loc[width.between(0.005, 0.04), "support_quality"] += 20
    out.loc[width.between(0.005, 0.04), "resistance_quality"] += 20
    out.loc[out["vol_ratio"] >= 1.1, ["support_quality", "resistance_quality"]] += 15
    out.loc[out["adx_14"] >= 18, ["support_quality", "resistance_quality"]] += 10
    out.loc[out["near_support"] & (out["close_location"] >= 0.5), "support_quality"] += 20
    out.loc[out["near_resistance"] & (out["close_location"] <= 0.5), "resistance_quality"] += 20
    out["support_quality"] = out["support_quality"].clip(0, 100)
    out["resistance_quality"] = out["resistance_quality"].clip(0, 100)
    return out


def detect_breakout_pullback(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    out = df.copy()
    out["breakout_signal"] = (
        (out["Close"] > out["resistance_20"]) &
        (out["Volume"] > out["vol_ma_20"] * 1.15) &
        (out["close_location"] >= 0.6)
    )
    out["pullback_signal"] = (
        (out["near_support"]) &
        (out["Close"] > out["sma_20"]) &
        (out["rsi_14"] >= cfg["rsi_floor"]) &
        (out["macd"] >= out["macd_signal"])
    )
    out["setup_type"] = np.select(
        [out["breakout_signal"], out["pullback_signal"]],
        ["BREAKOUT", "PULLBACK"],
        default="NONE"
    )
    return out


def detect_fake_breakouts(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["fake_breakout"] = False
    breakout_idx = out.index[out["breakout_signal"] == True].tolist()
    for idx in breakout_idx:
        pos = out.index.get_loc(idx)
        old_res = out.loc[idx, "resistance_20"]
        future = out.iloc[pos+1:pos+1+window]
        if len(future) and (future["Close"] < old_res).any():
            out.loc[idx, "fake_breakout"] = True
    return out


def enrich_price_action(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    out = compute_support_resistance(df, cfg["sr_lookback"])
    out = score_support_resistance_quality(out)
    out = detect_breakout_pullback(out, cfg)
    out = detect_fake_breakouts(out, cfg["fake_breakout_window"])
    out.dropna(inplace=True)
    return out

# =========================
# DATA CLASSES
# =========================
@dataclass
class TradePlan:
    action: str
    entry: float
    target: float
    stop: float
    risk_pct: float
    reward_pct: float
    rr_ratio: float
    shares: int
    position_value: float
    thesis: str
    invalidation: str
    setup_type: str


@dataclass
class AgentDecision:
    symbol: str
    timestamp: str
    price: float
    score: float
    action: str
    confidence: str
    trend_state: str
    market_state: str
    reasons: List[str]
    warnings: List[str]
    trade_plan: Optional[Dict]
    fundamentals: Dict
    latest_signals: Dict
    timeframe_summary: Dict
    setup_analytics: Dict

# =========================
# DECISION ASSISTANT
# =========================
class MultiTimeframeDecisionAssistant:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def _trend_state(self, row: pd.Series) -> str:
        if row["Close"] > row["sma_50"] > row["sma_200"]:
            return "Strong Uptrend"
        elif row["Close"] > row["sma_50"] and row["sma_50"] >= row["sma_200"]:
            return "Moderate Uptrend"
        elif row["Close"] < row["sma_50"] < row["sma_200"]:
            return "Strong Downtrend"
        return "Range / Mixed"

    def _market_state(self, row: pd.Series) -> str:
        if row["Close"] > row["sma_50"] > row["sma_200"]:
            return "Supportive"
        elif row["Close"] < row["sma_50"] < row["sma_200"]:
            return "Weak"
        return "Neutral"

    def _score_single(self, row: pd.Series, market_state: str) -> Tuple[float, List[str], List[str]]:
        score = 0.0
        reasons, warnings = [], []

        if row["Close"] > row["sma_20"]:
            score += 8; reasons.append("السعر أعلى من SMA20")
        else:
            warnings.append("السعر دون SMA20")

        if row["Close"] > row["sma_50"]:
            score += 10; reasons.append("السعر أعلى من SMA50")
        else:
            warnings.append("السعر دون SMA50")

        if row["sma_50"] > row["sma_200"]:
            score += 12; reasons.append("SMA50 أعلى من SMA200")
        else:
            warnings.append("البنية المتوسطة/الطويلة ليست صاعدة بالكامل")

        if self.cfg["rsi_floor"] <= row["rsi_14"] <= self.cfg["rsi_ceiling"]:
            score += 12; reasons.append("RSI في نطاق شراء صحي")
        elif row["rsi_14"] < 40:
            warnings.append("RSI ضعيف")
        elif row["rsi_14"] > 70:
            warnings.append("RSI متشبع نسبيًا")

        if row["macd"] > row["macd_signal"]:
            score += 10; reasons.append("MACD إيجابي")
        else:
            warnings.append("MACD سلبي")

        if row["adx_14"] >= 20:
            score += 8; reasons.append("ADX يشير لاتجاه مقبول")
        else:
            warnings.append("الاتجاه ضعيف نسبيًا")

        if row["vol_ratio"] >= 1.15:
            score += 8; reasons.append("الحجم أعلى من المعتاد")
        elif row["vol_ratio"] < 0.85:
            warnings.append("الحجم دون الطبيعي")

        if "relative_strength" in row and row["relative_strength"] > 0:
            score += 10; reasons.append("السهم يتفوق على السوق")
        else:
            warnings.append("السهم لا يتفوق على السوق مؤخرًا")

        if row["close_location"] >= 0.60:
            score += 6; reasons.append("الإغلاق قرب أعلى الشمعة")

        if 0.003 <= row["atr_pct"] <= 0.02:
            score += 6; reasons.append("التذبذب مناسب لهدف 1%")
        else:
            warnings.append("التذبذب الحالي قد لا يكون مثاليًا لهدف 1%")

        if row.get("setup_type", "NONE") == "BREAKOUT":
            if not bool(row.get("fake_breakout", False)):
                score += 8; reasons.append("هناك إشارة اختراق مدعومة")
            else:
                score -= 8; warnings.append("احتمال اختراق كاذب")
        elif row.get("setup_type", "NONE") == "PULLBACK":
            score += 8; reasons.append("هناك ارتداد من دعم")

        if row.get("support_quality", 0) >= 50:
            score += 5; reasons.append("جودة الدعم جيدة")
        if row.get("resistance_quality", 0) >= 50:
            score += 3; reasons.append("جودة المقاومة واضحة")

        if market_state == "Supportive":
            score += 5; reasons.append("السوق داعم")
        elif market_state == "Weak":
            score -= 5; warnings.append("السوق العام ضعيف")

        return max(0, min(100, score)), reasons, warnings

    def _combine_scores(self, daily_score: float, hourly_score: float, m15_score: float) -> float:
        return (0.45 * daily_score) + (0.35 * hourly_score) + (0.20 * m15_score)

    def _confidence_label(self, score: float) -> str:
        if score >= 80: return "High"
        if score >= self.cfg["min_score_buy"]: return "Medium"
        if score >= self.cfg["min_score_wait"]: return "Low"
        return "Very Low"

    def _build_trade_plan(self, row: pd.Series) -> Optional[TradePlan]:
        entry = float(row["Close"])
        target = entry * (1 + self.cfg["target_return"])
        atr_based = min(float(row["atr_pct"]) * 0.8, self.cfg["max_stop_pct"])
        risk_pct = max(0.0035, atr_based)
        stop = entry * (1 - risk_pct)
        reward_pct = (target / entry) - 1
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else np.nan

        max_capital_risk = self.cfg["capital"] * self.cfg["risk_per_trade"]
        risk_per_share = entry - stop
        shares = int(max_capital_risk // risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * entry
        if shares <= 0:
            return None

        return TradePlan(
            action="BUY",
            entry=entry,
            target=target,
            stop=stop,
            risk_pct=risk_pct,
            reward_pct=reward_pct,
            rr_ratio=rr_ratio,
            shares=shares,
            position_value=position_value,
            thesis="الصفقة صالحة فقط إذا استمر الزخم على الإطار التنفيذي.",
            invalidation="إلغاء الصفقة عند كسر وقف الخسارة أو ظهور اختراق كاذب أو فقدان SMA20.",
            setup_type=str(row.get("setup_type", "NONE")),
        )

    def decide(self, stock_daily, market_daily, stock_hourly, market_hourly, stock_15m, market_15m, fundamentals: Dict) -> AgentDecision:
        d_row = stock_daily.iloc[-1]
        d_mrow = market_daily.iloc[-1]
        h_row = stock_hourly.iloc[-1]
        h_mrow = market_hourly.iloc[-1]
        m_row = stock_15m.iloc[-1]
        m_mrow = market_15m.iloc[-1]

        d_market_state = self._market_state(d_mrow)
        h_market_state = self._market_state(h_mrow)
        m_market_state = self._market_state(m_mrow)

        d_score, d_reasons, d_warnings = self._score_single(d_row, d_market_state)
        h_score, h_reasons, h_warnings = self._score_single(h_row, h_market_state)
        m_score, m_reasons, m_warnings = self._score_single(m_row, m_market_state)

        final_score = self._combine_scores(d_score, h_score, m_score)
        trend_state = self._trend_state(d_row)
        market_state = d_market_state

        reasons = [f"[Daily] {x}" for x in d_reasons[:5]] + [f"[60m] {x}" for x in h_reasons[:5]] + [f"[15m] {x}" for x in m_reasons[:4]]
        warnings = [f"[Daily] {x}" for x in d_warnings[:4]] + [f"[60m] {x}" for x in h_warnings[:4]] + [f"[15m] {x}" for x in m_warnings[:4]]

        hard_block = False
        if trend_state == "Strong Downtrend":
            warnings.append("الاتجاه اليومي هابط بوضوح")
            hard_block = True
        if h_row["Close"] < h_row["sma_20"] and m_row["Close"] < m_row["sma_20"]:
            warnings.append("الإطاران التنفيذيان دون المتوسط القصير")
            hard_block = True
        if bool(m_row.get("fake_breakout", False)):
            warnings.append("آخر إشارة اختراق تبدو كاذبة")
            hard_block = True

        trade_plan = None
        if not hard_block and final_score >= self.cfg["min_score_buy"]:
            trade_plan = self._build_trade_plan(m_row)
            if trade_plan is None or trade_plan.rr_ratio < 1.3:
                action = "WAIT"
                warnings.append("خطة الصفقة لا تعطي نسبة عائد/مخاطرة كافية")
                trade_plan = None
            else:
                action = "BUY"
        elif final_score >= self.cfg["min_score_wait"]:
            action = "WAIT"
        else:
            action = "REDUCE / AVOID"

        latest_signals = {
            "daily_close": float(d_row["Close"]),
            "hourly_close": float(h_row["Close"]),
            "m15_close": float(m_row["Close"]),
            "setup_type_15m": str(m_row.get("setup_type", "NONE")),
            "fake_breakout_15m": bool(m_row.get("fake_breakout", False)),
            "support_20_15m": float(m_row.get("support_20", np.nan)),
            "resistance_20_15m": float(m_row.get("resistance_20", np.nan)),
            "support_quality_15m": float(m_row.get("support_quality", np.nan)),
            "resistance_quality_15m": float(m_row.get("resistance_quality", np.nan)),
        }

        timeframe_summary = {
            "daily_score": round(d_score, 2),
            "hourly_score": round(h_score, 2),
            "m15_score": round(m_score, 2),
            "daily_trend": self._trend_state(d_row),
            "hourly_trend": self._trend_state(h_row),
            "m15_trend": self._trend_state(m_row),
        }

        setup_analytics = {
            "current_setup": str(m_row.get("setup_type", "NONE")),
            "fake_breakout": bool(m_row.get("fake_breakout", False)),
            "support_quality": float(m_row.get("support_quality", np.nan)),
            "resistance_quality": float(m_row.get("resistance_quality", np.nan)),
        }

        return AgentDecision(
            symbol=fundamentals.get("symbol", "UNKNOWN"),
            timestamp=str(stock_15m.index[-1]),
            price=float(m_row["Close"]),
            score=round(final_score, 2),
            action=action,
            confidence=self._confidence_label(final_score),
            trend_state=trend_state,
            market_state=market_state,
            reasons=reasons,
            warnings=warnings,
            trade_plan=asdict(trade_plan) if trade_plan else None,
            fundamentals=fundamentals,
            latest_signals=latest_signals,
            timeframe_summary=timeframe_summary,
            setup_analytics=setup_analytics,
        )

# =========================
# BACKTEST
# =========================
class MultiTimeframeBacktester:
    def __init__(self, assistant: MultiTimeframeDecisionAssistant, hold_bars: int = 12):
        self.assistant = assistant
        self.hold_bars = hold_bars

    def run(self, stock_daily, market_daily, stock_hourly, market_hourly, stock_15m, market_15m) -> pd.DataFrame:
        rows = []
        idx = stock_15m.index
        for i in range(250, len(stock_15m) - self.hold_bars):
            ts = idx[i]
            sub_m15 = stock_15m.iloc[:i+1].copy()
            sub_market_m15 = market_15m.reindex(sub_m15.index).dropna()
            if len(sub_market_m15) < len(sub_m15) * 0.8:
                continue
            sub_m15 = sub_m15.loc[sub_market_m15.index]
            sub_hourly = stock_hourly.loc[stock_hourly.index <= ts].copy()
            sub_market_hourly = market_hourly.loc[market_hourly.index <= ts].copy()
            sub_daily = stock_daily.loc[stock_daily.index <= ts].copy()
            sub_market_daily = market_daily.loc[market_daily.index <= ts].copy()
            if min(len(sub_daily), len(sub_hourly), len(sub_m15)) < 220:
                continue
            decision = self.assistant.decide(sub_daily, sub_market_daily, sub_hourly, sub_market_hourly, sub_m15, sub_market_m15, fundamentals={"symbol": "BT"})
            if decision.action != "BUY" or decision.trade_plan is None:
                continue

            entry = decision.trade_plan["entry"]
            target = decision.trade_plan["target"]
            stop = decision.trade_plan["stop"]
            setup_type = decision.trade_plan.get("setup_type", "NONE")
            future = stock_15m.iloc[i+1:i+1+self.hold_bars]
            outcome = "TIME_EXIT"
            exit_price = future["Close"].iloc[-1]
            exit_time = future.index[-1]
            for t2, r in future.iterrows():
                hit_target = r["High"] >= target
                hit_stop = r["Low"] <= stop
                if hit_target and hit_stop:
                    outcome = "STOP"; exit_price = stop; exit_time = t2; break
                elif hit_target:
                    outcome = "TARGET"; exit_price = target; exit_time = t2; break
                elif hit_stop:
                    outcome = "STOP"; exit_price = stop; exit_time = t2; break

            rows.append({
                "entry_time": ts,
                "exit_time": exit_time,
                "entry": entry,
                "exit": exit_price,
                "score": decision.score,
                "setup_type": setup_type,
                "outcome": outcome,
                "return": (exit_price / entry) - 1,
                "hold_bars": len(future),
            })
        return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if bt.empty:
        return pd.DataFrame({"Metric": ["No trades"], "Value": ["No BUY signals"]}), pd.DataFrame()
    summary = pd.DataFrame({
        "Metric": ["Total Trades", "Target Hit Rate", "Stop Hit Rate", "Average Return", "Median Return"],
        "Value": [
            len(bt),
            f"{((bt['outcome'] == 'TARGET').mean())*100:.2f}%",
            f"{((bt['outcome'] == 'STOP').mean())*100:.2f}%",
            f"{bt['return'].mean()*100:.2f}%",
            f"{bt['return'].median()*100:.2f}%",
        ],
    })
    setup = bt.groupby("setup_type").agg(
        trades=("setup_type", "count"),
        win_rate=("outcome", lambda s: (s == "TARGET").mean()),
        avg_return=("return", "mean")
    ).reset_index()
    if not setup.empty:
        setup["win_rate"] = setup["win_rate"].map(lambda x: f"{x*100:.2f}%")
        setup["avg_return"] = setup["avg_return"].map(lambda x: f"{x*100:.2f}%")
    return summary, setup

# =========================
# PIPELINE
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def run_analysis(stock_ticker: str, market_ticker: str, cfg: Dict):
    loader = MarketDataLoader(stock_ticker, market_ticker)
    raw = loader.load_all(cfg)
    fundamentals = loader.get_info_snapshot()

    stock_daily = FeatureEngineer.add_features(raw["stock_daily"], raw["market_daily"], cfg["relative_strength_window"])
    market_daily = FeatureEngineer.add_features(raw["market_daily"], None, cfg["relative_strength_window"])
    stock_hourly = FeatureEngineer.add_features(raw["stock_hourly"], raw["market_hourly"], cfg["relative_strength_window"])
    market_hourly = FeatureEngineer.add_features(raw["market_hourly"], None, cfg["relative_strength_window"])
    stock_15m = FeatureEngineer.add_features(raw["stock_15m"], raw["market_15m"], cfg["relative_strength_window"])
    market_15m = FeatureEngineer.add_features(raw["market_15m"], None, cfg["relative_strength_window"])

    market_daily = market_daily.reindex(stock_daily.index).dropna()
    stock_daily = stock_daily.loc[market_daily.index]
    market_hourly = market_hourly.reindex(stock_hourly.index).dropna()
    stock_hourly = stock_hourly.loc[market_hourly.index]
    market_15m = market_15m.reindex(stock_15m.index).dropna()
    stock_15m = stock_15m.loc[market_15m.index]

    stock_daily = enrich_price_action(stock_daily, cfg)
    stock_hourly = enrich_price_action(stock_hourly, cfg)
    stock_15m = enrich_price_action(stock_15m, cfg)

    assistant = MultiTimeframeDecisionAssistant(cfg)
    decision = assistant.decide(stock_daily, market_daily, stock_hourly, market_hourly, stock_15m, market_15m, fundamentals)

    bt_engine = MultiTimeframeBacktester(assistant, hold_bars=cfg["backtest_hold_bars"])
    bt = bt_engine.run(stock_daily, market_daily, stock_hourly, market_hourly, stock_15m, market_15m)
    bt_summary, setup_summary = summarize_backtest(bt)

    return {
        "decision": decision,
        "backtest": bt,
        "backtest_summary": bt_summary,
        "setup_summary": setup_summary,
        "daily": stock_daily,
        "hourly": stock_hourly,
        "m15": stock_15m,
    }

# =========================
# CHARTS
# =========================
def build_dashboard(stock_daily: pd.DataFrame, stock_hourly: pd.DataFrame, stock_15m: pd.DataFrame):
    d = stock_daily.tail(120).copy()
    h = stock_hourly.tail(120).copy()
    m = stock_15m.tail(120).copy()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.06,
                        subplot_titles=("Daily", "60 Minutes", "15 Minutes"))

    for row_idx, df, label in [(1, d, "Daily"), (2, h, "60m"), (3, m, "15m")]:
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=f"{label} Price"), row=row_idx, col=1)
        for ma in ["sma_20", "sma_50", "sma_200"]:
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode="lines", name=f"{label} {ma}"), row=row_idx, col=1)
        if "support_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["support_20"], mode="lines", line=dict(dash="dot"), name=f"{label} support"), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["resistance_20"], mode="lines", line=dict(dash="dot"), name=f"{label} resistance"), row=row_idx, col=1)

    fig.update_layout(height=1100, title="Multi-Timeframe Dashboard", xaxis_rangeslider_visible=False)
    return fig

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Saudi Stock Decision Assistant", layout="wide")
st.title("Saudi Stock Decision Assistant")
st.caption("أدخل رمز السهم مثل 7010.SR أو 7203.SR للحصول على التحليل")

with st.sidebar:
    st.header("الإعدادات")
    stock_ticker = st.text_input("رمز السهم", value="7010.SR")
    market_ticker = st.text_input("رمز السوق المرجعي", value=DEFAULT_MARKET_TICKER)
    capital = st.number_input("رأس المال", min_value=1000, value=100000, step=1000)
    risk_per_trade = st.slider("المخاطرة لكل صفقة", 0.001, 0.05, 0.01, 0.001)
    target_return = st.slider("هدف الصفقة", 0.002, 0.03, 0.01, 0.001)
    max_stop_pct = st.slider("أقصى وقف خسارة", 0.002, 0.03, 0.006, 0.001)
    run_btn = st.button("تحليل السهم", type="primary")

cfg = DEFAULT_CONFIG.copy()
cfg["capital"] = capital
cfg["risk_per_trade"] = risk_per_trade
cfg["target_return"] = target_return
cfg["max_stop_pct"] = max_stop_pct

if run_btn:
    try:
        with st.spinner("جاري التحليل..."):
            results = run_analysis(stock_ticker.strip(), market_ticker.strip(), cfg)
        decision = results["decision"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("القرار", decision.action)
        c2.metric("السكور", f"{decision.score:.2f}")
        c3.metric("السعر", f"{decision.price:.2f}")
        c4.metric("الثقة", decision.confidence)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["الملخص", "خطة الصفقة", "الرسوم", "الاختبار التاريخي", "البيانات الأساسية"])

        with tab1:
            st.subheader("الملخص التنفيذي")
            st.write({
                "symbol": decision.symbol,
                "timestamp": decision.timestamp,
                "trend_state": decision.trend_state,
                "market_state": decision.market_state,
                "setup": decision.setup_analytics,
                "timeframes": decision.timeframe_summary,
            })
            st.markdown("**الأسباب**")
            for r in decision.reasons:
                st.write(f"- {r}")
            st.markdown("**التحذيرات**")
            for w in decision.warnings:
                st.write(f"- {w}")

        with tab2:
            st.subheader("خطة الصفقة")
            if decision.trade_plan:
                st.json(decision.trade_plan)
            else:
                st.info("لا توجد صفقة نشطة حاليًا لأن الشروط لم تكتمل.")

        with tab3:
            st.subheader("لوحة الرسوم")
            fig = build_dashboard(results["daily"], results["hourly"], results["m15"])
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("ملخص الاختبار التاريخي")
            st.dataframe(results["backtest_summary"], use_container_width=True)
            st.subheader("أداء أنواع الإعداد")
            st.dataframe(results["setup_summary"], use_container_width=True)
            if not results["backtest"].empty:
                st.subheader("آخر 10 صفقات")
                st.dataframe(results["backtest"].tail(10), use_container_width=True)

        with tab5:
            st.subheader("البيانات الأساسية")
            st.json(decision.fundamentals)

    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل: {e}")
else:
    st.info("أدخل رمز السهم من القائمة الجانبية ثم اضغط: تحليل السهم")
