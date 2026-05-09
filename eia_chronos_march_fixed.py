import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import torch

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 8  # Jan–Aug 2025
    model_id: str = "amazon/chronos-t5-tiny"

def load_config(config_path=None) -> 'Config':
    """Build Config from config.yaml, falling back to dataclass defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return Config()
    with open(config_path) as _f:
        import yaml as _yaml
        raw = _yaml.safe_load(_f) or {}
    _d = raw.get('data', {})
    _m = raw.get('model', {})
    _o = raw.get('output', {})
    return Config(
        csv_path=_d.get('input_file', '2001-2025 Net_generation_United_States_all_sectors_monthly.csv'),
        freq=_d.get('freq', 'MS'),
        horizon=_m.get('horizon', 8),
        model_id=_m.get('model_id', 'amazon/chronos-t5-tiny'),
    )



def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def main():
    cfg = load_config()
    y = load_series(cfg)

    end_2024 = pd.Timestamp("2024-12-01")
    jan_2025 = pd.Timestamp("2025-01-01")
    aug_2025 = pd.Timestamp("2025-08-01")

    y_train = y.loc[:end_2024]
    y_act = y.loc[jan_2025:aug_2025]

    # Try ChronosPipeline (chronos-forecasting exposes the same name)
    try:
        from chronos import ChronosPipeline
    except Exception:
        # Some installs expose under chronos_forecasting
        from chronos_forecasting import ChronosPipeline  # type: ignore

    pipe = ChronosPipeline.from_pretrained(cfg.model_id, device_map="cpu")
    # Chronos expects a numpy array
    arr = y_train.values.astype(np.float32)
    context = torch.tensor(arr, dtype=torch.float32, device="cpu")
    out = pipe.predict(context, prediction_length=cfg.horizon)
    out_np = np.asarray(out)
    # If multiple samples were returned, average across samples
    out_np = np.where(out_np.size % cfg.horizon == 0, out_np.reshape(-1, cfg.horizon).mean(axis=0), out_np.ravel()[:cfg.horizon])
    dates = pd.period_range("2025-01", "2025-08", freq="M").to_timestamp()
    fc = pd.Series(out_np, index=dates)

    # Greyscale Tufte-style plot
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    ax.plot(fc.index, fc.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")

    if len(y_hist):
        ax.annotate(
            "History (2024)",
            xy=(y_hist.index[-1], y_hist.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#666666",
        )
    if len(y_act):
        ax.annotate(
            "Actual (Jan–Aug 2025)",
            xy=(y_act.index[-1], y_act.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#444444",
        )
    ax.annotate(
        "Chronos",
        xy=(fc.index[-1], fc.values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        ha="left",
        color="#000000",
    )

    ax.set_title("EIA Net Generation — Chronos forecast Jan–Aug 2025")
    save_fig("eia_chronos_last_fold.png")


if __name__ == "__main__":
    main()
