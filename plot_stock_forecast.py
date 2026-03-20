from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


FORECAST_DIR = Path("forecast_output")
OUTPUT_DIR = Path("forecast_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full stock forecast history for one symbol using all generated forecast_output files."
    )
    parser.add_argument("symbol", type=str, help="Ticker symbol, e.g. A or AAPL")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="forecast_output",
        help="Folder containing generated forecast CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="forecast_plots",
        help="Folder where plots will be saved",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots after saving",
    )
    return parser.parse_args()


def load_symbol_rows(csv_path: Path, symbol: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame()

    out = df[df["Symbol"].astype(str).str.upper() == symbol.upper()].copy()
    return out


def collect_test_predictions(forecast_dir: Path, symbol: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for frequency in ["daily", "weekly", "monthly"]:
        path = forecast_dir / f"{frequency}_test_predictions.csv"
        df = load_symbol_rows(path, symbol)
        if df.empty:
            continue

        df["SourceFrequency"] = frequency
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    if "ForecastDate" in df_all.columns:
        df_all["ForecastDate"] = pd.to_datetime(df_all["ForecastDate"], errors="coerce")
    if "NextClose" in df_all.columns:
        df_all["NextClose"] = pd.to_numeric(df_all["NextClose"], errors="coerce")
    if "PredictedNextClose" in df_all.columns:
        df_all["PredictedNextClose"] = pd.to_numeric(df_all["PredictedNextClose"], errors="coerce")

    df_all = df_all.dropna(subset=["ForecastDate"])
    df_all = df_all.sort_values("ForecastDate").reset_index(drop=True)
    return df_all


def collect_next_forecasts(forecast_dir: Path, symbol: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for frequency in ["daily", "weekly", "monthly"]:
        path = forecast_dir / f"{frequency}_next_forecasts.csv"
        df = load_symbol_rows(path, symbol)
        if df.empty:
            continue

        df["SourceFrequency"] = frequency
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    if "ForecastDate" in df_all.columns:
        df_all["ForecastDate"] = pd.to_datetime(df_all["ForecastDate"], errors="coerce")
    if "PredictedNextClose" in df_all.columns:
        df_all["PredictedNextClose"] = pd.to_numeric(df_all["PredictedNextClose"], errors="coerce")

    df_all = df_all.dropna(subset=["ForecastDate"])
    df_all = df_all.sort_values("ForecastDate").reset_index(drop=True)
    return df_all


def build_actual_series(df_test: pd.DataFrame) -> pd.DataFrame:
    if df_test.empty:
        return pd.DataFrame(columns=["ForecastDate", "ActualClose"])

    actual = df_test[["ForecastDate", "NextClose"]].copy()
    actual = actual.rename(columns={"NextClose": "ActualClose"})
    actual = actual.dropna(subset=["ForecastDate", "ActualClose"])

    actual = (
        actual.groupby("ForecastDate", as_index=False)["ActualClose"]
        .mean()
        .sort_values("ForecastDate")
        .reset_index(drop=True)
    )
    return actual


def build_predicted_series(df_test: pd.DataFrame) -> pd.DataFrame:
    if df_test.empty:
        return pd.DataFrame(columns=["ForecastDate", "PredictedClose"])

    predicted = df_test[["ForecastDate", "PredictedNextClose"]].copy()
    predicted = predicted.rename(columns={"PredictedNextClose": "PredictedClose"})
    predicted = predicted.dropna(subset=["ForecastDate", "PredictedClose"])

    predicted = (
        predicted.groupby("ForecastDate", as_index=False)["PredictedClose"]
        .mean()
        .sort_values("ForecastDate")
        .reset_index(drop=True)
    )
    return predicted


def plot_actual_history(actual_df: pd.DataFrame, symbol: str, output_dir: Path) -> Path:
    if actual_df.empty:
        raise ValueError(f"No actual historical rows found for symbol '{symbol}'.")

    plt.figure(figsize=(12, 6))
    plt.plot(actual_df["ForecastDate"], actual_df["ActualClose"], label="Actual Close")
    plt.title(f"{symbol} Actual Stock Movement (Whole Generated Timeline)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{symbol}_actual_whole.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_actual_vs_predicted(
    actual_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    next_df: pd.DataFrame,
    symbol: str,
    output_dir: Path,
) -> Path:
    if actual_df.empty:
        raise ValueError(f"No actual historical rows found for symbol '{symbol}'.")
    if predicted_df.empty:
        raise ValueError(f"No predicted historical rows found for symbol '{symbol}'.")

    plt.figure(figsize=(12, 6))
    plt.plot(actual_df["ForecastDate"], actual_df["ActualClose"], label="Actual Close")
    plt.plot(predicted_df["ForecastDate"], predicted_df["PredictedClose"], label="Predicted Close")

    if not next_df.empty:
        next_plot = next_df[["ForecastDate", "PredictedNextClose"]].copy()
        next_plot = next_plot.dropna(subset=["ForecastDate", "PredictedNextClose"])
        next_plot = (
            next_plot.groupby("ForecastDate", as_index=False)["PredictedNextClose"]
            .mean()
            .sort_values("ForecastDate")
        )

        if not next_plot.empty:
            plt.plot(
                next_plot["ForecastDate"],
                next_plot["PredictedNextClose"],
                marker="o",
                linestyle="None",
                label="Latest Saved Forecast",
            )

    plt.title(f"{symbol} Actual vs Model Prediction (Whole Generated Timeline)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{symbol}_predicted_vs_actual_whole.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper().strip()
    forecast_dir = Path(args.forecast_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_test = collect_test_predictions(forecast_dir, symbol)
    df_next = collect_next_forecasts(forecast_dir, symbol)

    if df_test.empty and df_next.empty:
        raise ValueError(
            f"Symbol '{symbol}' was not found in any generated forecast_output CSV files."
        )

    actual_df = build_actual_series(df_test)
    predicted_df = build_predicted_series(df_test)

    actual_plot = plot_actual_history(actual_df, symbol, output_dir)
    predicted_plot = plot_actual_vs_predicted(actual_df, predicted_df, df_next, symbol, output_dir)

    print(f"Saved actual plot to: {actual_plot}")
    print(f"Saved predicted-vs-actual plot to: {predicted_plot}")

    if args.show:
        actual_img = plt.imread(actual_plot)
        plt.figure(figsize=(12, 6))
        plt.imshow(actual_img)
        plt.axis("off")
        plt.title(actual_plot.name)

        pred_img = plt.imread(predicted_plot)
        plt.figure(figsize=(12, 6))
        plt.imshow(pred_img)
        plt.axis("off")
        plt.title(predicted_plot.name)
        plt.show()


if __name__ == "__main__":
    main()