---
author: "Kyle Jones"
date_published: "March 14, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/forecasting-ercot-load-with-amazon-chronos-in-python-cf8be7a81a75"
---

# Forecasting ERCOT Load with Amazon Chronos in Python

Amazon Chronos is a time series LLM. We will use it to predict energy demand in the ERCOT (Electric Reliability Council of Texas) power...

### Forecasting ERCOT Load with Amazon Chronos in Python 

Amazon Chronos is a time series LLM. We will use it to predict energy demand in the ERCOT (Electric Reliability Council of Texas) power grid. We'll walk through the process, including data preprocessing, training a model, and evaluating its performance.

Amazon Chronos is a family of pretrained time series forecasting models based on language model architectures. These models transform time series data into sequences of tokens using scaling and quantization.

Chronos models have been trained on a corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes, allowing them to generalize well across a wide range of time series forecasting tasks.

The models are based on the T5 architecture, with a few modifications. The main difference is in the vocabulary size: Chronos-T5 models use 4096 different tokens, compared to the 32128 tokens used by the original T5 models. This results in fewer parameters, making the models more efficient for time series tasks.

For a detailed explanation of Chronos models, training data, procedures, and experimental results, refer to the paper *Chronos: Learning the Language of Time Series*.


<figcaption>Image from Amazon Chronos</figcaption>


The target time series is scaled and quantized into a sequence of tokens.

These tokens are then fed into a language model, which may be an encoder-decoder or a decoder-only model. The model is trained using cross-entropy loss.

During inference, the model autoregressively samples tokens and maps them back to numerical values. Multiple trajectories are sampled to create a predictive distribution, allowing for probabilistic forecasting.

We begin by loading historical ERCOT load data, which provides the power consumption (load) over time. The data is stored in a CSV file, where each record corresponds to a timestamp and the corresponding load value.

Chronos is not available through pypi. To install it use: `pip install git+https://github.com/amazon-science/chronos-forecasting.git`

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from chronos import ChronosPipeline

# Load data
df = pd.read_csv("ercot_load_data.csv")
# Convert values to PyTorch tensor
context = torch.tensor(df["values"].values)
```

In this code, we load the data into a DataFrame, and we focus on the `values` column which contains the load data. We then convert the values into a PyTorch tensor, which is the format required by the Chronos model.

Since time series data has a temporal structure, it is important to split it while preserving this structure. We use TimeSeriesSplit from `sklearn` to perform a split that keeps the chronological order intact.

``` 
# Define prediction length
prediction_length = 96

# Split the data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=2, test_size=prediction_length)
for train_index, test_index in tscv.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]
# Extract training context
train_context = torch.tensor(train["values"].values)
```

Here, we define the `prediction_length` as 96, meaning that our model will forecast the next 96 time steps because the observations are every 15 mins (so 96 observations a day). Chronos will warn you that the model is not optimized for such a long prediction window.

Once the data is prepared, we load a pre-trained forecasting model from Amazon Chronos. This model is based on T5, a transformer-based architecture, and is ready for fine-tuning on time series data. Chronos does better with gpus and cuda. But I was able to run it with my laptop and Google Colab using cpus just fine.

``` 
# Load Chronos pipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Forecast
forecast = pipeline.predict(train_context, prediction_length)  
```

We use `ChronosPipeline.from_pretrained()` to load a pre-trained T5 model tailored for time series forecasting. The `predict` method generates forecasts for the next `prediction_length` steps based on the training context.

To evaluate the model's performance, we calculate the Mean Absolute Percentage Error (MAPE), which is a common metric for forecasting accuracy. We compare the predicted values (the median forecast) against the true values in the test set.

``` 
# Get forecast statistics
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# Compute MAPE
true_values = test["values"].values
mape = np.mean(np.abs((true_values - median) / true_values)) * 100
print(f"MAPE: {mape:.2f}%")


MAPE: 1.27%
```

The forecast output is an array of predicted values, from which we extract the 10th percentile (`low`), 50th percentile (`median`), and 90th percentile (`high`) to represent the range of uncertainty in the forecast. We then compute the MAPE, which gives us an idea of how close our predictions are to the true values, expressed as a percentage.

Finally, we visualize the forecasted values, historical data, and true values using Matplotlib.

``` 
# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["values"], color="royalblue", label="Historical data")
plt.plot(test.index, true_values, color="green", label="True values", linestyle="dashed")
plt.plot(test.index, median, color="red", label="Median forecast")
plt.fill_between(test.index, low, high, color="red", alpha=0.3, label="80% prediction interval")

plt.xlabel("Time")
plt.ylabel("Load")
plt.legend()
plt.title("ERCOT Load Forecast")
plt.show()
```


<figcaption>Chronos was able to model the fluctuations in this load very well.</figcaption>


Amazon Chronos did a great job forecasting this real-world dataset. The key difference between Chronos and traditional time series is that the model is pre-trained. We are not creating the model. In this pipeline, our task is inference only. That simplifies a lot of things but also gives up a lot of control. I did more experimenting with this using a larger dataset, Ercot hourly load from 2018--2025 (Feb). It did ok.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Load data
df = pd.read_csv("Ercot_Native_Load_2025 (1).csv")

# Ensure 'Date' column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Extract values
series = df["ERCOT"].values
dates = df["Date"].values

# Hold out the last 64 values
hold_out_length = 64
train_series = series[:-hold_out_length]  
actual_holdout = series[-hold_out_length:]

# Prepare Chronos input with only training data
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Convert series to tensor
context = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0) 
forecast = pipeline.predict(context, hold_out_length)  

# Extract forecast quantiles
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# Select dates for plotting (last 128 values: 64 before + 64 holdout)
plot_range = 64 * 2
plot_series = series[-plot_range:]  # Last 128 values
plot_dates = dates[-plot_range:]  # Corresponding dates

# Forecast dates (aligned to the last 64 hold-out values)
forecast_dates = dates[-hold_out_length:]

# Calculate MAPE
error = mape(actual_holdout, median)
print(f"MAPE: {error:.2f}%")

# Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(plot_dates, plot_series, color="black", label="Actual (Last 128)")
plt.plot(forecast_dates, median, color="red", linestyle="dashed", label="Forecast (64)")
plt.fill_between(forecast_dates, low, high, color="red", alpha=0.3, label="80% Prediction Interval")
plt.xticks(rotation=45, fontsize=10, fontname="serif")
plt.yticks(fontsize=10, fontname="serif")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_position(("outward", 10))
plt.gca().spines["bottom"].set_position(("outward", 10))
plt.legend(frameon=False, fontsize=10)
plt.savefig("ercot_forecast_vs_actual_with_dates.png")
plt.show()
```
I revisited Chronos with a different dataset. Below is the code using data about [US energy generation.](https://www.eia.gov/electricity/data/browser/)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import torch

np.random.seed(42)
plt.rcParams.update({
    'axes.grid': False,'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 8  # Jan–Aug 2025
    model_id: str = "amazon/chronos-t5-tiny"


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def main():
    cfg = Config()
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
    if out_np.size % cfg.horizon == 0:
        out_np = out_np.reshape(-1, cfg.horizon).mean(axis=0)
    else:
        out_np = out_np.ravel()[:cfg.horizon]
    dates = pd.period_range('2025-01', '2025-08', freq='M').to_timestamp()
    fc = pd.Series(out_np, index=dates)

    # Greyscale Tufte-style plot
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    ax.plot(fc.index, fc.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        ax.set_xlabel('')

    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666')
    if len(y_act):
        ax.annotate('Actual (Jan–Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444')
    ax.annotate('Chronos', xy=(fc.index[-1], fc.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000')

    ax.set_title('EIA Net Generation — Chronos forecast Jan–Aug 2025')
    save_fig('eia_chronos_last_fold.png')

if __name__ == '__main__':
    main()
```
