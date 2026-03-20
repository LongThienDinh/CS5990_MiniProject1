from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
LOOKBACKS = [1, 2, 3, 5, 10, 20]
ROLL_WINDOWS = [5, 10, 20]
MAX_ABS_FEATURE_VALUE = 50.0
MAX_ABS_TARGET_LOG_RETURN = 1.0
PROGRESS_EVERY_SYMBOLS = 250

PROGRESS_DRAW_DELTA = 0.02
PROGRESS_MIN_INTERVAL_SEC = 0.08


class ProgressPrinter:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._active_line = False
        self._last_line_length = 0
        self._epoch = -1
        self._last_draw_progress = -1.0
        self._last_draw_time = 0.0
        self._last_checkpoint = -1

    @staticmethod
    def _format_loss(loss_value: float) -> str:
        if math.isnan(loss_value):
            return "nan"
        return f"{loss_value:.6f}"

    @staticmethod
    def _build_bar(progress: float, width: int = 30) -> str:
        filled = int(width * progress)
        return "#" * filled + "-" * (width - filled)

    def log(self, message: str) -> None:
        if self._active_line:
            self.stream.write("\n")
            self._active_line = False
            self._last_line_length = 0
        self.stream.write(message + "\n")
        self.stream.flush()

    def start_epoch(self, epoch: int, epochs: int) -> None:
        self._epoch = epoch
        self._last_draw_progress = -1.0
        self._last_draw_time = 0.0
        self._last_checkpoint = -1
        self.update_epoch(epoch, epochs, 0.0, float("nan"))

    def update_epoch(self, epoch: int, epochs: int, progress: float, loss_value: float) -> None:
        progress = max(0.0, min(1.0, progress))

        if self.interactive:
            now = time.monotonic()
            should_draw = (
                epoch != self._epoch
                or progress >= 1.0
                or self._last_draw_progress < 0.0
                or (progress - self._last_draw_progress) >= PROGRESS_DRAW_DELTA
                or (now - self._last_draw_time) >= PROGRESS_MIN_INTERVAL_SEC
            )
            if not should_draw:
                return

            message = (
                f"Epoch {epoch:02d}/{epochs:02d} "
                f"[{self._build_bar(progress)}] {progress * 100:6.2f}% "
                f"loss={self._format_loss(loss_value)}"
            )
            pad = max(0, self._last_line_length - len(message))
            self.stream.write("\r" + message + (" " * pad))
            self.stream.flush()

            self._active_line = True
            self._last_line_length = len(message)
            self._last_draw_progress = progress
            self._last_draw_time = now
            self._epoch = epoch
            return

        checkpoint = int(progress * 20)
        should_print = (
            epoch != self._epoch
            or progress >= 1.0
            or self._last_checkpoint < 0
            or checkpoint > self._last_checkpoint
        )
        if should_print:
            self.stream.write(
                f"Epoch {epoch:02d}/{epochs:02d} "
                f"{progress * 100:6.2f}% "
                f"loss={self._format_loss(loss_value)}\n"
            )
            self.stream.flush()
            self._last_checkpoint = checkpoint
            self._last_draw_progress = progress
            self._epoch = epoch

    def finish_epoch(self, epoch: int, epochs: int, elapsed_seconds: float) -> None:
        if self._active_line:
            self.stream.write("\n")
            self.stream.flush()
            self._active_line = False
            self._last_line_length = 0
        self.log(f"Epoch {epoch:02d}/{epochs:02d} finished in {elapsed_seconds:.2f} seconds")


LOGGER = ProgressPrinter(sys.stdout)


def live_log(message: str) -> None:
    LOGGER.log(message)


def clip_numeric_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=MAX_ABS_FEATURE_VALUE,
        neginf=-MAX_ABS_FEATURE_VALUE,
    )
    return np.clip(X, -MAX_ABS_FEATURE_VALUE, MAX_ABS_FEATURE_VALUE)


def configure_runtime_warnings() -> None:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*(overflow encountered in matmul|divide by zero encountered in matmul|invalid value encountered in matmul).*",
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast next-period stock close using MLPRegressor for daily, weekly, and monthly data. "
            "Each .txt file is treated as one symbol/company."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .txt stock files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="forecast_output",
        help="Folder where models, metrics, and forecast CSV files are saved.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of each symbol series reserved for testing when possible.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_iter", type=int, default=3, help="Number of training epochs.")
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional cap on number of input files to load. 0 means all files.",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=100000,
        help="Optional cap on training rows per frequency for faster fitting. 0 means no cap.",
    )
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="32,16",
        help="Comma-separated hidden layer sizes, e.g. 64,32 or 32,16.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size used for the outer training loop and epoch progress bar.",
    )
    parser.add_argument(
        "--skip_log_limit",
        type=int,
        default=0,
        help="Maximum number of skip messages to print. Use 0 to suppress skip logs.",
    )
    return parser.parse_args()


def parse_hidden_layers(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("--hidden_layers must contain at least one integer")
    return tuple(int(p) for p in parts)


def infer_symbol(path: Path) -> str:
    stem = path.stem
    if "." in stem:
        return stem.rsplit(".", 1)[0].upper()
    return stem.upper()


def _try_read_csv(path: Path, sep: str | None) -> pd.DataFrame:
    kwargs = {
        "engine": "python",
        "on_bad_lines": "skip",
    }
    kwargs["sep"] = sep if sep is not None else None
    return pd.read_csv(path, **kwargs)


def load_stock_file(path: Path) -> pd.DataFrame:
    read_attempts: List[Tuple[str, str | None]] = [("comma", ","), ("tab", "\t"), ("auto", None)]

    last_error: Exception | None = None
    df: pd.DataFrame | None = None

    for _, sep in read_attempts:
        try:
            candidate = _try_read_csv(path, sep)
            if all(col in candidate.columns for col in REQUIRED_COLUMNS):
                df = candidate
                break
        except Exception as exc:
            last_error = exc

    if df is None:
        if last_error is not None:
            raise ValueError(f"unable to read expected columns ({last_error})")
        raise ValueError("missing required columns")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
    df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
    df["Symbol"] = infer_symbol(path)
    return df


def load_all_files(
    input_dir: Path,
    max_files: int,
    skip_log_limit: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    files = sorted(list(input_dir.glob("*.txt")) + list(input_dir.glob("*.TXT")))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    data: Dict[str, pd.DataFrame] = {}
    skipped_records: List[Dict[str, str]] = []
    printed = 0

    live_log(f"Loading stock files from {input_dir} ...")

    for idx, path in enumerate(files, start=1):
        try:
            df = load_stock_file(path)
            if df.empty:
                reason = "no usable rows after cleaning"
                skipped_records.append({"file": path.name, "reason": reason})
                if skip_log_limit > 0 and printed < skip_log_limit:
                    live_log(f"Skipping {path.name}: {reason}")
                    printed += 1
                continue
            data[df["Symbol"].iloc[0]] = df
        except Exception as exc:
            reason = str(exc)
            skipped_records.append({"file": path.name, "reason": reason})
            if skip_log_limit > 0 and printed < skip_log_limit:
                live_log(f"Skipping {path.name}: {reason}")
                printed += 1

        if idx % 1000 == 0 or idx == len(files):
            live_log(f"Loaded file scan progress: {idx}/{len(files)}")

    if skip_log_limit > 0 and printed >= skip_log_limit and len(skipped_records) > skip_log_limit:
        live_log(f"... plus {len(skipped_records) - skip_log_limit} more skipped files during loading")

    if not data:
        raise RuntimeError("No valid stock files were loaded.")

    return data, pd.DataFrame(skipped_records)


def resample_ohlcv(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "daily":
        return df.copy().reset_index(drop=True)

    frame = df.copy().set_index("Date")
    agg_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "OpenInt": "last",
        "Symbol": "last",
    }

    if frequency == "weekly":
        out = frame.resample("W-FRI").agg(agg_map)
    elif frequency == "monthly":
        out = frame.resample("ME").agg(agg_map)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    return out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index()


def add_calendar_features(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = out["Date"].dt.year.astype(int)
    out["Month"] = out["Date"].dt.month.astype(int)
    out["Quarter"] = out["Date"].dt.quarter.astype(int)
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["DayOfWeek"] = out["Date"].dt.dayofweek.astype(int)
    out["IsMonthEnd"] = out["Date"].dt.is_month_end.astype(int)
    out["IsQuarterEnd"] = out["Date"].dt.is_quarter_end.astype(int)
    out["FrequencyCode"] = {"daily": 0, "weekly": 1, "monthly": 2}[frequency]
    return out


def _clip_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    return s.clip(lower=lower, upper=upper)


def feature_columns() -> List[str]:
    cols = [
        "Year",
        "Month",
        "Quarter",
        "WeekOfYear",
        "DayOfWeek",
        "IsMonthEnd",
        "IsQuarterEnd",
        "FrequencyCode",
        "LogClose",
        "LogVolume",
        "IntradayReturn",
        "HighCloseSpread",
        "LowCloseSpread",
        "LogReturn_1",
        "LogVolumeChange_1",
    ]
    for lb in LOOKBACKS:
        cols.extend([f"LogReturn_{lb}", f"LogVolumeChange_{lb}", f"LogCloseLag_{lb}", f"LogVolumeLag_{lb}"])
    for win in ROLL_WINDOWS:
        cols.extend([f"CloseToSMA_{win}", f"ReturnSTD_{win}", f"VolumeMean_{win}", f"VolumeSTD_{win}"])
    return cols


def build_features(df: pd.DataFrame, frequency: str, include_target: bool) -> pd.DataFrame:
    out = add_calendar_features(df, frequency)

    close = out["Close"]
    open_ = out["Open"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"].clip(lower=0)
    log_close = np.log(close)
    log_volume = np.log1p(volume)

    out["LogClose"] = log_close
    out["LogVolume"] = log_volume
    out["IntradayReturn"] = _clip_series((close / open_) - 1.0, -2.0, 2.0)
    out["HighCloseSpread"] = _clip_series((high / close) - 1.0, -2.0, 2.0)
    out["LowCloseSpread"] = _clip_series((low / close) - 1.0, -2.0, 2.0)
    out["LogReturn_1"] = _clip_series(log_close.diff(1), -2.0, 2.0)
    out["LogVolumeChange_1"] = _clip_series(log_volume.diff(1), -5.0, 5.0)

    for lb in LOOKBACKS:
        out[f"LogReturn_{lb}"] = _clip_series(log_close.diff(lb), -2.0, 2.0)
        out[f"LogVolumeChange_{lb}"] = _clip_series(log_volume.diff(lb), -5.0, 5.0)
        out[f"LogCloseLag_{lb}"] = log_close.shift(lb)
        out[f"LogVolumeLag_{lb}"] = log_volume.shift(lb)

    for win in ROLL_WINDOWS:
        sma = close.rolling(win).mean()
        ret_std = out["LogReturn_1"].rolling(win).std(ddof=0)
        out[f"CloseToSMA_{win}"] = _clip_series((close / sma) - 1.0, -2.0, 2.0)
        out[f"ReturnSTD_{win}"] = _clip_series(ret_std, 0.0, 5.0)
        out[f"VolumeMean_{win}"] = log_volume.rolling(win).mean()
        out[f"VolumeSTD_{win}"] = _clip_series(log_volume.rolling(win).std(ddof=0), 0.0, 10.0)

    if include_target:
        out["TargetLogReturn"] = _clip_series(
            np.log(out["Close"].shift(-1) / out["Close"]),
            -MAX_ABS_TARGET_LOG_RETURN,
            MAX_ABS_TARGET_LOG_RETURN,
        )
        out["NextClose"] = out["Close"].shift(-1)
        out["ForecastDate"] = out["Date"].shift(-1)

    out = out.replace([np.inf, -np.inf], np.nan)

    if include_target:
        out = out.dropna(subset=["Date", "Close", "TargetLogReturn", "NextClose", "ForecastDate"]).reset_index(drop=True)
    else:
        out = out.dropna(subset=["Date", "Close"]).reset_index(drop=True)

    return out


def next_period_date(last_date: pd.Timestamp, frequency: str) -> pd.Timestamp:
    if frequency == "daily":
        return last_date + BDay(1)
    if frequency == "weekly":
        days_to_friday = (4 - last_date.weekday()) % 7
        return last_date + pd.Timedelta(days=7 if days_to_friday == 0 else days_to_friday)
    if frequency == "monthly":
        return last_date + MonthEnd(1)
    raise ValueError(f"Unsupported frequency: {frequency}")


def prepare_frequency_dataset(
    all_data: Dict[str, pd.DataFrame],
    frequency: str,
    skip_log_limit: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainable_frames: List[pd.DataFrame] = []
    latest_feature_rows: List[pd.DataFrame] = []
    skipped_records: List[Dict[str, str]] = []
    printed = 0
    total_symbols = len(all_data)

    live_log(f"{frequency}: dataset preparation started for {total_symbols} symbols")

    for idx, (symbol, raw_df) in enumerate(all_data.items(), start=1):
        resampled = resample_ohlcv(raw_df, frequency)

        if resampled.empty:
            reason = "no rows after resampling"
            skipped_records.append({"symbol": symbol, "reason": reason})
            if skip_log_limit > 0 and printed < skip_log_limit:
                live_log(f"Skipping {symbol} for {frequency}: {reason}")
                printed += 1
            if idx % PROGRESS_EVERY_SYMBOLS == 0 or idx == total_symbols:
                live_log(f"{frequency}: prepared {idx}/{total_symbols} symbols")
            continue

        with_target = build_features(resampled, frequency, include_target=True)
        without_target = build_features(resampled, frequency, include_target=False)

        if not without_target.empty:
            latest_row = without_target.tail(1).copy()
            latest_row["ForecastDate"] = next_period_date(latest_row["Date"].iloc[0], frequency)
            latest_feature_rows.append(latest_row)

        if with_target.empty:
            reason = "no target rows available for training"
            skipped_records.append({"symbol": symbol, "reason": reason})
            if skip_log_limit > 0 and printed < skip_log_limit:
                live_log(f"Skipping {symbol} for {frequency}: {reason}")
                printed += 1
            if idx % PROGRESS_EVERY_SYMBOLS == 0 or idx == total_symbols:
                live_log(f"{frequency}: prepared {idx}/{total_symbols} symbols")
            continue

        trainable_frames.append(with_target)

        if idx % PROGRESS_EVERY_SYMBOLS == 0 or idx == total_symbols:
            live_log(f"{frequency}: prepared {idx}/{total_symbols} symbols")

    if skip_log_limit > 0 and printed >= skip_log_limit and len(skipped_records) > skip_log_limit:
        live_log(f"... plus {len(skipped_records) - skip_log_limit} more skipped symbols for {frequency}")

    if not trainable_frames:
        raise RuntimeError(f"No usable series for frequency={frequency}")

    live_log(f"{frequency}: concatenating trainable frames ...")
    combined_train = pd.concat(trainable_frames, ignore_index=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)

    if latest_feature_rows:
        live_log(f"{frequency}: concatenating latest rows ...")
        combined_latest = pd.concat(latest_feature_rows, ignore_index=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    else:
        combined_latest = pd.DataFrame()

    live_log(
        f"{frequency}: dataset preparation finished "
        f"(train_rows={len(combined_train)}, latest_rows={len(combined_latest)})"
    )

    return combined_train, combined_latest, pd.DataFrame(skipped_records)


def time_split_by_symbol(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in df.groupby("Symbol", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)

        if n <= 1:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)
        split_idx = n - n_test

        train_part = group.iloc[:split_idx]
        test_part = group.iloc[split_idx:]

        if not train_part.empty:
            train_parts.append(train_part)
        if not test_part.empty:
            test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train_df, test_df


def build_preprocessor() -> Pipeline:
    clipper = FunctionTransformer(
        clip_numeric_array,
        validate=False,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper_before_scale", clipper),
            ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
            ("clipper_after_scale", clipper),
        ]
    )


def fit_mlp_with_progress(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    hidden_layers: Tuple[int, ...],
    random_state: int,
    epochs: int,
    batch_size: int = 256,
) -> Tuple[Pipeline, StandardScaler, MLPRegressor]:
    n_samples_raw = X_train_raw.shape[0]
    if n_samples_raw == 0:
        raise ValueError("X_train_raw is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    live_log(f"Preparing training data: {n_samples_raw} rows")

    t0 = time.time()
    preprocessor = build_preprocessor()
    live_log("Preprocessing started ...")
    X_train = preprocessor.fit_transform(X_train_raw)
    live_log(f"Preprocessing finished in {time.time() - t0:.2f} seconds")

    t1 = time.time()
    target_scaler = StandardScaler()
    live_log("Target scaling started ...")
    y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    live_log(f"Target scaling finished in {time.time() - t1:.2f} seconds")

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate_init=3e-4,
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=random_state,
        verbose=False,
    )

    n_samples = X_train.shape[0]
    total_steps = int(np.ceil(n_samples / batch_size))
    rng = np.random.default_rng(random_state)

    live_log(f"Training started: {epochs} epochs, {total_steps} steps per epoch")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        indices = rng.permutation(n_samples)
        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        LOGGER.start_epoch(epoch, epochs)

        for step, start in enumerate(range(0, n_samples, batch_size), start=1):
            end = min(start + batch_size, n_samples)
            X_batch = X_epoch[start:end]
            y_batch = y_epoch[start:end]

            mlp.partial_fit(X_batch, y_batch)

            progress = step / total_steps
            loss_value = getattr(mlp, "loss_", float("nan"))
            LOGGER.update_epoch(epoch, epochs, progress, loss_value)

        LOGGER.finish_epoch(epoch, epochs, time.time() - epoch_start)

    return preprocessor, target_scaler, mlp


def predict_log_returns(
    preprocessor: Pipeline,
    target_scaler: StandardScaler,
    mlp: MLPRegressor,
    X_raw: np.ndarray,
) -> np.ndarray:
    X = preprocessor.transform(X_raw)
    pred_scaled = mlp.predict(X)
    pred_log_returns = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return np.clip(pred_log_returns, -MAX_ABS_TARGET_LOG_RETURN, MAX_ABS_TARGET_LOG_RETURN)


def maybe_downsample_train(train_df: pd.DataFrame, max_train_rows: int, random_state: int) -> pd.DataFrame:
    if max_train_rows <= 0 or len(train_df) <= max_train_rows:
        return train_df
    return train_df.sample(n=max_train_rows, random_state=random_state).sort_values(["Symbol", "Date"]).reset_index(drop=True)


def safe_log_return_to_price(last_close: np.ndarray, pred_log_return: np.ndarray) -> np.ndarray:
    pred_log_return = np.clip(pred_log_return, -MAX_ABS_TARGET_LOG_RETURN, MAX_ABS_TARGET_LOG_RETURN)
    return last_close * np.exp(pred_log_return)


def evaluate_predictions(test_df: pd.DataFrame, pred_log_returns: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred_log_returns = np.clip(
        np.asarray(pred_log_returns, dtype=np.float64),
        -MAX_ABS_TARGET_LOG_RETURN,
        MAX_ABS_TARGET_LOG_RETURN,
    )

    eval_df = test_df[["Symbol", "Date", "ForecastDate", "Close", "NextClose", "TargetLogReturn"]].copy()
    eval_df["PredictedLogReturn"] = pred_log_returns
    eval_df["PredictedReturn"] = np.exp(eval_df["PredictedLogReturn"]) - 1.0
    eval_df["ActualReturn"] = np.exp(eval_df["TargetLogReturn"]) - 1.0
    eval_df["PredictedNextClose"] = safe_log_return_to_price(
        eval_df["Close"].to_numpy(dtype=np.float64),
        pred_log_returns,
    )
    eval_df["AbsoluteError"] = (eval_df["NextClose"] - eval_df["PredictedNextClose"]).abs()
    eval_df["PctError"] = eval_df["AbsoluteError"] / eval_df["NextClose"].replace(0, np.nan)

    y_true_close = eval_df["NextClose"].to_numpy(dtype=np.float64)
    y_pred_close = eval_df["PredictedNextClose"].to_numpy(dtype=np.float64)
    y_true_log = eval_df["TargetLogReturn"].to_numpy(dtype=np.float64)
    y_pred_log = eval_df["PredictedLogReturn"].to_numpy(dtype=np.float64)

    metrics = {
        "samples": int(len(eval_df)),
        "mae_close": float(mean_absolute_error(y_true_close, y_pred_close)),
        "rmse_close": float(np.sqrt(mean_squared_error(y_true_close, y_pred_close))),
        "r2_close": float(r2_score(y_true_close, y_pred_close)),
        "mape_close": float(np.nanmean(eval_df["PctError"].to_numpy(dtype=np.float64)) * 100.0),
        "mae_log_return": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log_return": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log_return": float(r2_score(y_true_log, y_pred_log)),
    }
    return eval_df, metrics


def empty_metrics() -> Dict[str, float]:
    return {
        "samples": 0,
        "mae_close": float("nan"),
        "rmse_close": float("nan"),
        "r2_close": float("nan"),
        "mape_close": float("nan"),
        "mae_log_return": float("nan"),
        "rmse_log_return": float("nan"),
        "r2_log_return": float("nan"),
    }


def train_for_frequency(
    all_data: Dict[str, pd.DataFrame],
    frequency: str,
    output_dir: Path,
    test_size: float,
    random_state: int,
    max_iter: int,
    max_train_rows: int,
    hidden_layers: Tuple[int, ...],
    batch_size: int,
    skip_log_limit: int,
) -> Dict[str, float]:
    live_log(f"\n=== Training frequency: {frequency} ===")

    dataset, latest_rows, skipped_df = prepare_frequency_dataset(
        all_data=all_data,
        frequency=frequency,
        skip_log_limit=skip_log_limit,
    )

    skipped_df.to_csv(output_dir / f"{frequency}_skipped_symbols.csv", index=False)

    live_log(f"{frequency}: time split started ...")
    train_df, test_df = time_split_by_symbol(dataset, test_size=test_size)
    original_train_rows = len(train_df)
    train_df = maybe_downsample_train(train_df, max_train_rows=max_train_rows, random_state=random_state)

    if train_df.empty:
        raise RuntimeError(f"No training rows available for frequency={frequency}")

    cols = feature_columns()
    X_train_raw = clip_numeric_array(train_df[cols].to_numpy(dtype=np.float64))
    y_train_raw = np.clip(
        train_df["TargetLogReturn"].to_numpy(dtype=np.float64),
        -MAX_ABS_TARGET_LOG_RETURN,
        MAX_ABS_TARGET_LOG_RETURN,
    )

    live_log(
        f"{frequency}: symbols_used={dataset['Symbol'].nunique()}, "
        f"train_rows={len(train_df)} (before cap={original_train_rows}), test_rows={len(test_df)}"
    )
    live_log(f"{frequency}: training for {max_iter} epochs ...")

    preprocessor, target_scaler, mlp = fit_mlp_with_progress(
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
        hidden_layers=hidden_layers,
        random_state=random_state,
        epochs=max_iter,
        batch_size=batch_size,
    )

    if not test_df.empty:
        live_log(f"{frequency}: evaluating test predictions ...")
        X_test_raw = clip_numeric_array(test_df[cols].to_numpy(dtype=np.float64))
        pred_log_returns = predict_log_returns(
            preprocessor=preprocessor,
            target_scaler=target_scaler,
            mlp=mlp,
            X_raw=X_test_raw,
        )
        eval_df, metrics = evaluate_predictions(test_df, pred_log_returns)
        eval_df.insert(0, "Frequency", frequency)
        eval_df.to_csv(output_dir / f"{frequency}_test_predictions.csv", index=False)
    else:
        metrics = empty_metrics()
        pd.DataFrame().to_csv(output_dir / f"{frequency}_test_predictions.csv", index=False)

    if not latest_rows.empty:
        live_log(f"{frequency}: generating next forecasts ...")
        latest_X_raw = clip_numeric_array(latest_rows[cols].to_numpy(dtype=np.float64))
        latest_pred_log_returns = predict_log_returns(
            preprocessor=preprocessor,
            target_scaler=target_scaler,
            mlp=mlp,
            X_raw=latest_X_raw,
        )

        forecasts = latest_rows[["Symbol", "Date", "ForecastDate", "Close"]].copy()
        forecasts.insert(0, "Frequency", frequency)
        forecasts["PredictedLogReturn"] = latest_pred_log_returns
        forecasts["PredictedReturn"] = np.exp(forecasts["PredictedLogReturn"]) - 1.0
        forecasts["PredictedNextClose"] = safe_log_return_to_price(
            forecasts["Close"].to_numpy(dtype=np.float64),
            latest_pred_log_returns,
        )
        forecasts.rename(columns={"Date": "LastObservedDate", "Close": "LastClose"}, inplace=True)
        forecasts.to_csv(output_dir / f"{frequency}_next_forecasts.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / f"{frequency}_next_forecasts.csv", index=False)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "target_scaler": target_scaler,
            "model": mlp,
            "feature_columns": cols,
            "frequency": frequency,
            "metrics": metrics,
            "hidden_layers": hidden_layers,
            "target": "TargetLogReturn",
        },
        output_dir / f"{frequency}_mlp_model.joblib",
    )

    metrics_row = {
        "frequency": frequency,
        **metrics,
        "train_rows": int(len(train_df)),
        "train_rows_before_cap": int(original_train_rows),
        "test_rows": int(len(test_df)),
        "symbols_used": int(dataset["Symbol"].nunique()),
        "symbols_skipped": int(len(skipped_df)),
    }

    with open(output_dir / f"{frequency}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_row, f, indent=2)

    live_log(
        f"{frequency.capitalize()} done -> "
        f"MAE(close)={metrics['mae_close']}, "
        f"RMSE(close)={metrics['rmse_close']}, "
        f"R2(close)={metrics['r2_close']}, "
        f"MAPE(close)={metrics['mape_close']}"
    )
    return metrics_row


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)

    configure_runtime_warnings()
    args = parse_args()
    hidden_layers = parse_hidden_layers(args.hidden_layers)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data, skipped_load_df = load_all_files(
        input_dir=input_dir,
        max_files=args.max_files,
        skip_log_limit=args.skip_log_limit,
    )
    skipped_load_df.to_csv(output_dir / "loading_skipped_files.csv", index=False)

    live_log(f"Loaded {len(all_data)} valid symbols from {input_dir}")

    summary_rows: List[Dict[str, float]] = []
    all_forecasts: List[pd.DataFrame] = []

    for frequency in ["daily", "weekly", "monthly"]:
        try:
            metrics_row = train_for_frequency(
                all_data=all_data,
                frequency=frequency,
                output_dir=output_dir,
                test_size=args.test_size,
                random_state=args.random_state,
                max_iter=args.max_iter,
                max_train_rows=args.max_train_rows,
                hidden_layers=hidden_layers,
                batch_size=args.batch_size,
                skip_log_limit=args.skip_log_limit,
            )
            summary_rows.append(metrics_row)
            forecast_path = output_dir / f"{frequency}_next_forecasts.csv"
            if forecast_path.exists():
                df_forecast = pd.read_csv(forecast_path)
                if not df_forecast.empty:
                    all_forecasts.append(df_forecast)
        except Exception as exc:
            live_log(f"Could not complete {frequency}: {exc}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
        live_log(f"Saved metrics summary to {output_dir / 'metrics_summary.csv'}")

    if all_forecasts:
        pd.concat(all_forecasts, ignore_index=True).to_csv(output_dir / "all_next_forecasts.csv", index=False)
        live_log(f"Saved combined forecasts to {output_dir / 'all_next_forecasts.csv'}")

    live_log("Done.")


if __name__ == "__main__":
    main()