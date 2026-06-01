# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# AUTO sys.path FIX
# ------------------------------------------------------------------
_THIS_FILE    = Path(__file__).resolve()
_SRC_DIR      = _THIS_FILE.parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
_EXPECTED_COLS = ["Open", "High", "Low", "Close", "Volume", "Price"]
_DATA_REL_PATH = Path("data") / "raw" / "btc_usd.csv"
_HISTORY_START = "2014-09-17"


# ==================================================================
class BitcoinDataLoader:

    def __init__(self) -> None:
        self.project_root: Path = _PROJECT_ROOT
        self.data_path: Path    = self.project_root / _DATA_REL_PATH
        log.info("Project root : %s", self.project_root)
        log.info("CSV path     : %s", self.data_path)

    # --------------------------------------------------------------
    # PUBLIC
    # --------------------------------------------------------------

    def update(self) -> pd.DataFrame:
        log.info("=== BTC update started ===")

        local_df   = self._load_local()
        start_date = self._resolve_start_date(local_df)

        if start_date is None:
            log.info("CSV is already up-to-date.")
            return local_df if local_df is not None else pd.DataFrame(columns=_EXPECTED_COLS)

        # Çoklu kaynak denemesi (Waterfall/Şelale Mantığı)
        patch = self._fetch_with_fallback(start_date)

        if patch is None or patch.empty:
            log.error("ALL SOURCES FAILED. Check internet or API status.")
            return local_df if local_df is not None else pd.DataFrame(columns=_EXPECTED_COLS)

        merged = self._smart_merge(local_df, patch)
        merged = self._validate(merged)

        if merged is not None and not merged.empty:
            self._safe_save(merged)
            log.info("=== Update complete: %d rows ===", len(merged))
            return merged

        log.error("Validation failed. CSV not modified.")
        return local_df if local_df is not None else pd.DataFrame(columns=_EXPECTED_COLS)

    # --------------------------------------------------------------
    # PRIVATE — LOCAL
    # --------------------------------------------------------------

    def _load_local(self) -> pd.DataFrame | None:
        if not self.data_path.exists():
            log.info("No local CSV found.")
            return None
        try:
            df = pd.read_csv(
                self.data_path,
                parse_dates=["Date"],
                index_col="Date",
            )
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            log.info("CSV loaded: %d rows  (%s to %s)",
                     len(df), df.index.min().date(), df.index.max().date())
            return df
        except Exception as exc:
            log.error("Failed to read CSV: %s", exc)
            return None

    def _resolve_start_date(self, local_df: pd.DataFrame | None) -> str | None:
        yesterday = pd.Timestamp(
            datetime.now(tz=timezone.utc).date() - timedelta(days=1)
        ).tz_localize(None)

        if local_df is None:
            log.info("Full history will be fetched from: %s", _HISTORY_START)
            return _HISTORY_START

        last_date = local_df.index.max()
        if last_date >= yesterday:
            return None

        start = last_date + timedelta(days=1)
        log.info("Missing range: %s to %s", start.date(), yesterday.date())
        return start.strftime("%Y-%m-%d")

    # --------------------------------------------------------------
    # PRIVATE — FETCH (MULTIPLE SOURCES)
    # --------------------------------------------------------------

    def _fetch_with_fallback(self, start_date: str) -> pd.DataFrame | None:
        """Sırasıyla kaynakları dener, biri başarılı olursa döner."""
        
        # 1. Deneme: Binance (En stabil ve hızlı)
        patch = self._fetch_binance(start_date)
        if patch is not None and not patch.empty:
            return patch

        # 2. Deneme: yfinance (Binance başarısız olursa)
        log.warning("Binance failed or no data -> trying yfinance.")
        patch = self._fetch_yfinance(start_date)
        if patch is not None and not patch.empty:
            return patch

        # 3. Deneme: CoinGecko (Son çare)
        log.warning("yfinance failed -> trying CoinGecko.")
        return self._fetch_coingecko(start_date)

    def _fetch_binance(self, start_date: str) -> pd.DataFrame | None:
        try:
            log.info("Calling Binance API...")
            url      = "https://api.binance.com/api/v3/klines"
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            all_rows: list[pd.DataFrame] = []

            while True:
                params = {
                    "symbol"   : "BTCUSDT",
                    "interval" : "1d",
                    "startTime": start_ts,
                    "limit"    : 1000,
                }
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                if not isinstance(data, list) or len(data) == 0:
                    break

                chunk = pd.DataFrame(data, columns=[
                    "open_time","Open","High","Low","Close","Volume",
                    "close_time","quote_vol","trades","taker_base",
                    "taker_quote","ignore",
                ])
                chunk["Date"] = pd.to_datetime(chunk["open_time"], unit="ms")
                chunk = chunk.set_index("Date")
                chunk = chunk[["Open","High","Low","Close","Volume"]].astype(float)
                chunk.index = chunk.index.tz_localize(None)
                all_rows.append(chunk)

                last_close_ts = int(data[-1][6])
                if len(data) < 1000:
                    break
                start_ts = last_close_ts + 1

            if not all_rows:
                return None

            df = pd.concat(all_rows)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df["Price"] = df["Close"]
            log.info("Binance: %d rows fetched.", len(df))
            return df
        except Exception as exc:
            log.warning("Binance error: %s", exc)
            return None

    def _fetch_yfinance(self, start_date: str) -> pd.DataFrame | None:
        try:
            log.info("Calling yfinance API...")
            raw = yf.download(
                "BTC-USD",
                start=start_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if raw is None or raw.empty:
                return None

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]

            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df["Price"] = df["Close"]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            log.info("yfinance: %d rows fetched.", len(df))
            return df
        except Exception as exc:
            log.warning("yfinance error: %s", exc)
            return None

    def _fetch_coingecko(self, start_date: str) -> pd.DataFrame | None:
        try:
            log.info("Calling CoinGecko API...")
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts   = int(datetime.now().timestamp())
            
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
            params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
            
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "prices" not in data or not data["prices"]:
                return None

            df = pd.DataFrame(data["prices"], columns=["Date", "Price"])
            df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.normalize()
            df = df.set_index("Date")
            
            # OHLCV verisi CoinGecko basit API'sinde kısıtlıdır, Price ile dolduruyoruz
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = df["Price"]
            df["Volume"] = 0.0 # Hacim verisi farklı listede olduğu için 0 bırakıyoruz
                
            df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            log.info("CoinGecko: %d rows fetched.", len(df))
            return df
        except Exception as exc:
            log.warning("CoinGecko error: %s", exc)
            return None

    # --------------------------------------------------------------
    # PRIVATE — MERGE / VALIDATE / SAVE
    # --------------------------------------------------------------

    def _smart_merge(self, local: pd.DataFrame | None, patch: pd.DataFrame) -> pd.DataFrame:
        if local is None:
            merged = patch.copy()
        else:
            # Sadece yerel dosyada olmayan tarihleri ekle
            new_dates = patch.index.difference(local.index)
            if new_dates.empty:
                log.info("No new dates to add.")
                return local
            merged = pd.concat([local, patch.loc[new_dates]])

        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        
        # Kolon kontrolü
        for col in _EXPECTED_COLS:
            if col not in merged.columns:
                merged[col] = float("nan")
        
        return merged[_EXPECTED_COLS]

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            log.error("Validation: DataFrame is empty.")
            return None

        # Negatif fiyat kontrolü
        price_cols = ["Open","High","Low","Close","Price"]
        if (df[price_cols] < 0).any().any():
            log.error("Validation: Negative price values detected.")
            return None

        # Mantıksal High/Low kontrolü
        if not (df["High"] >= df["Low"]).all():
            df = df[df["High"] >= df["Low"]]

        # Kayıp veri oranı kontrolü (%20'den fazlası boşsa kaydetme)
        nan_ratio = df.isna().mean().mean()
        if nan_ratio > 0.20:
            log.error("Validation: Too many NaNs (%.1f%%).", nan_ratio * 100)
            return None

        return df

    def _safe_save(self, df: pd.DataFrame) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", dir=self.data_path.parent, delete=False, encoding="utf-8"
            ) as tmp:
                tmp_path = Path(tmp.name)
                df.to_csv(tmp, index_label="Date")

            shutil.move(str(tmp_path), str(self.data_path))
            log.info("CSV updated successfully: %s", self.data_path)
        except Exception as exc:
            log.error("CSV save error: %s", exc)
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            raise