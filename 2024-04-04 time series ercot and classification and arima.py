"""Generated from Jupyter notebook: 2024-04-04 time series ercot and classification and arima

Magics and shell lines are commented out. Run with a normal Python interpreter."""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


def simulated_time_series_data() -> None:
    np.random.seed(42)

    data = pd.Series(np.cumsum(np.random.randn(200)))

    df = pd.DataFrame({'value': data, 'lag_1': data.shift(1), 'lag_2': data.shift(2), 'rate_of_change': data.diff()}).dropna()

    df['direction'] = (df['value'].shift(-1) > df['value']).astype(int)

    df['next_value'] = df['value'].shift(-1)

    df = df.dropna()

    X = df[['value', 'lag_1', 'lag_2', 'rate_of_change']]

    y_class = df['direction']

    y_reg = df['next_value']

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train, y_class_train)

    X_train_reg = X_train.copy()

    X_test_reg = X_test.copy()

    X_train_reg['direction_pred'] = clf.predict(X_train)

    X_test_reg['direction_pred'] = clf.predict(X_test)

    reg = RandomForestRegressor(random_state=42)

    reg.fit(X_train_reg, y_reg_train)

    y_class_pred = clf.predict(X_test)

    y_reg_pred = reg.predict(X_test_reg)

    accuracy = accuracy_score(y_class_test, y_class_pred)

    mae = mean_absolute_error(y_reg_test, y_reg_pred)

    print(f'Classification Accuracy: {accuracy:.2f}')

    print(f'Regression Mean Absolute Error: {mae:.2f}')

    train_index = pd.date_range(start='2000-01-01', periods=len(y_reg_train), freq='D')

    y_reg_train_series = pd.Series(y_reg_train.values, index=train_index)

    p = d = q = range(0, 3)

    pdq = list(itertools.product(p, d, q))

    y_reg_train_series = y_reg_train_series.diff().dropna()

    best_aic = np.inf

    best_pdq = None

    best_model = None

    for param in pdq:
        try:
            model_arima = ARIMA(y_reg_train_series, order=param)
            results_arima = model_arima.fit()
            if results_arima.aic < best_aic:
                best_aic = results_arima.aic
                best_pdq = param
                best_model = results_arima
        except Exception as e:
            print(f'ARIMA{param} failed: {e}')
            continue

    if best_model is not None:
        print(f'Best ARIMA model: ARIMA{best_pdq}')
        forecast = best_model.get_forecast(steps=len(y_reg_test))
        y_reg_pred_arima = forecast.predicted_mean
        mae_arima = mean_absolute_error(y_reg_test, y_reg_pred_arima)
        print(f'Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}')
    else:
        print('No valid ARIMA model found.')

    print(f'Best ARIMA model: ARIMA{best_pdq}')

    forecast_index = pd.date_range(start=train_index[-1] + pd.Timedelta(days=1), periods=len(y_reg_test), freq='D')

    forecast = best_model.forecast(steps=len(y_reg_test))

    y_reg_pred_arima = forecast

    accuracy_rf = accuracy_score(y_class_test, clf.predict(X_test))

    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred_rf)

    mae_arima = mean_absolute_error(y_reg_test, y_reg_pred_arima)

    print(f'Classification Accuracy (Random Forest): {accuracy_rf:.2f}')

    print(f'Regression Mean Absolute Error (Random Forest): {mae_rf:.2f}')

    print(f'Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}')


def ensure_data_stationarity_optional_use_differenci() -> None:
    y_reg_train_series = y_reg_train_series.diff().dropna()

    best_aic = np.inf

    best_pdq = None

    best_model = None

    for param in pdq:
        try:
            model_arima = ARIMA(y_reg_train_series, order=param)
            results_arima = model_arima.fit()
            if results_arima.aic < best_aic:
                best_aic = results_arima.aic
                best_pdq = param
                best_model = results_arima
        except Exception as e:
            print(f'ARIMA{param} failed: {e}')
            continue

    if best_model is not None:
        print(f'Best ARIMA model: ARIMA{best_pdq}')
        forecast = best_model.get_forecast(steps=len(y_reg_test))
        y_reg_pred_arima = forecast.predicted_mean
        mae_arima = mean_absolute_error(y_reg_test, y_reg_pred_arima)
        print(f'Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}')
    else:
        print('No valid ARIMA model found.')


def suppress_arima_warnings() -> None:
    warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found')

    warnings.filterwarnings('ignore', category=UserWarning)

    y_reg_train_series = y_reg_train_series.diff().dropna()

    adf_test = adfuller(y_reg_train_series)

    print(f'ADF Statistic: {adf_test[0]}')

    print(f'p-value: {adf_test[1]}')

    if adf_test[1] > 0.05:
        print('Warning: Time series is not stationary!')

    best_aic = np.inf

    best_pdq = None

    best_model = None

    for param in pdq:
        try:
            model_arima = ARIMA(y_reg_train_series, order=param)
            results_arima = model_arima.fit()
            if results_arima.aic < best_aic:
                best_aic = results_arima.aic
                best_pdq = param
                best_model = results_arima
        except Exception as e:
            print(f'ARIMA{param} failed: {e}')
            continue

    if best_model is not None:
        print(f'Best ARIMA model: ARIMA{best_pdq}')
        forecast = best_model.get_forecast(steps=len(y_reg_test))
        y_reg_pred_arima = forecast.predicted_mean
        mae_arima = mean_absolute_error(y_reg_test, y_reg_pred_arima)
        print(f'Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}')
    else:
        print('No valid ARIMA model found.')

    accuracy_rf = accuracy_score(y_class_test, clf.predict(X_test))

    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred)

    print(f'Classification Accuracy (Random Forest): {accuracy_rf:.2f}')

    print(f'Regression Mean Absolute Error (Random Forest): {mae_rf:.2f}')


def suppress_arima_warnings_2() -> None:
    warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found')

    warnings.filterwarnings('ignore', category=UserWarning)

    np.random.seed(42)

    data = pd.Series(np.cumsum(np.random.randn(200)))

    df = pd.DataFrame({'value': data, 'lag_1': data.shift(1), 'lag_2': data.shift(2), 'rate_of_change': data.diff()}).dropna()

    df['direction'] = (df['value'].shift(-1) > df['value']).astype(int)

    df['next_value'] = df['value'].shift(-1)

    df = df.dropna()

    X = df[['value', 'lag_1', 'lag_2', 'rate_of_change']]

    y_class = df['direction']

    y_reg = df['next_value']

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train, y_class_train)

    X_train_reg = X_train.copy()

    X_test_reg = X_test.copy()

    X_train_reg['direction_pred'] = clf.predict(X_train)

    X_test_reg['direction_pred'] = clf.predict(X_test)

    reg = RandomForestRegressor(random_state=42)

    reg.fit(X_train_reg, y_reg_train)

    y_class_pred = clf.predict(X_test)

    y_reg_pred = reg.predict(X_test_reg)

    accuracy = accuracy_score(y_class_test, y_class_pred)

    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred)

    plt.figure(figsize=(10, 6))

    plt.plot(df['value'], label='Original Time Series')

    plt.title('Original Time Series')

    plt.legend()

    plt.savefig('original_time_series.png')

    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(y_reg_test.values, label='Actual Values', color='Blue')

    plt.plot(y_reg_pred, label='Random Forest Predictions', color='Red')

    plt.title('Random Forest Predictions vs Actual Values')

    plt.legend()

    plt.savefig('random_forest_predictions.png')

    plt.show()

    residuals_rf = y_reg_test.values - y_reg_pred

    plt.figure(figsize=(10, 6))

    plt.plot(residuals_rf, label='Random Forest Residuals', color='Blue')

    plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')

    plt.title('Random Forest Residuals')

    plt.legend()

    plt.savefig('random_forest_residuals.png')

    plt.show()

    y_reg_train_series = y_reg_train.diff().dropna()

    adf_test = adfuller(y_reg_train_series)

    print(f'ADF Statistic: {adf_test[0]}')

    print(f'p-value: {adf_test[1]}')

    if adf_test[1] > 0.05:
        print('Warning: Time series is not stationary!')

    p = d = q = range(0, 3)

    pdq = list(itertools.product(p, d, q))

    best_aic = np.inf

    best_pdq = None

    best_model = None

    for param in pdq:
        try:
            model_arima = ARIMA(y_reg_train_series, order=param)
            results_arima = model_arima.fit()
            if results_arima.aic < best_aic:
                best_aic = results_arima.aic
                best_pdq = param
                best_model = results_arima
        except:
            continue

    if best_model is not None:
        print(f'Best ARIMA model: ARIMA{best_pdq}')
        forecast = best_model.get_forecast(steps=len(y_reg_test))
        y_reg_pred_arima_diff = forecast.predicted_mean
        last_value = y_reg_train.iloc[-1]
        y_reg_pred_arima = np.cumsum(y_reg_pred_arima_diff) + last_value
        mae_arima = mean_absolute_error(y_reg_test, y_reg_pred_arima)
        print(f'Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}')
        residuals_arima = y_reg_test.values - y_reg_pred_arima
        plt.figure(figsize=(10, 6))
        plt.plot(residuals_arima, label='ARIMA Residuals', color='Blue')
        plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
        plt.title('ARIMA Residuals')
        plt.legend()
        plt.savefig('arima_residuals.png')
        plt.show()
    else:
        print('No valid ARIMA model found.')

    print(f'Classification Accuracy (Random Forest): {accuracy:.2f}')

    print(f'Regression Mean Absolute Error (Random Forest): {mae_rf:.2f}')

    if best_model is not None:
        print(f'Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}')


def suppress_warnings() -> None:
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    # Suppress warnings
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the ERCOT data
    df = pd.read_csv("ercot_load_data.csv")
    df["date"] = pd.to_datetime(df["date"])  # Ensure 'date' is in datetime format
    df["values"] = pd.to_numeric(
        df["values"], errors="coerce"
    )  # Convert 'values' to numeric
    df = df.sort_values("date")  # Sort by date
    df = df.dropna()  # Drop rows with missing or NaN values

    # Resample the data to hourly frequency
    df = (
        df.set_index("date").resample("h").mean().reset_index()
    )  # Resample and take the mean for each hour

    # Define hold-out period
    hold_out_hours = 24  # 24 hours = 1 day
    train = df.iloc[:-hold_out_hours]
    hold_out = df.iloc[-hold_out_hours:]

    # Generate Lagged Features
    df["lag_1"] = df["values"].shift(1)
    df["lag_2"] = df["values"].shift(2)
    df["rate_of_change"] = df["values"].diff()

    # Target for Classification: 1 if increase, 0 if decrease
    df["direction"] = (df["values"].shift(-1) > df["values"]).astype(int)

    # Target for Regression: Next Value
    df["next_value"] = df["values"].shift(-1)

    # Drop NaN Rows (after generating lag features)
    df = df.dropna()

    # Train-Test Split
    X = df[["values", "lag_1", "lag_2", "rate_of_change"]]
    y_class = df["direction"]
    y_reg = df["next_value"]
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)
    )

    # Classification Model: Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_class_train)

    # Add Classification Predictions as a Feature for Regression
    X_train_reg = X_train.copy()
    X_test_reg = X_test.copy()
    X_train_reg["direction_pred"] = clf.predict(X_train)
    X_test_reg["direction_pred"] = clf.predict(X_test)

    # Regression Model: Random Forest Regressor
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_reg, y_reg_train)

    # Predict Values
    y_class_pred = clf.predict(X_test)
    y_reg_pred = reg.predict(X_test_reg)

    # Evaluate Performance
    accuracy = accuracy_score(y_class_test, y_class_pred)
    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred)

    # Plot Original Time Series
    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["values"], label="Original Time Series")
    plt.title("ERCOT Load Data - Original Time Series")
    plt.xlabel("Date")
    plt.ylabel("Load Values")
    plt.legend()
    plt.savefig("ercot_original_time_series.png")
    plt.show()

    # Plot Predictions vs Actual (Random Forest)
    plt.figure(figsize=(10, 6))
    plt.plot(y_reg_test.values, label="Actual Values", color="Blue")
    plt.plot(y_reg_pred, label="Random Forest Predictions", color="Red")
    plt.title("Random Forest Predictions vs Actual Values")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig("ercot_rf_predictions.png")
    plt.show()

    # ARIMA Model
    y_reg_train_series = y_reg_train.diff().dropna()

    # Check Stationarity (Optional)
    adf_test = adfuller(y_reg_train_series)
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    if adf_test[1] > 0.05:
        print("Warning: Time series is not stationary!")

    # Grid Search for ARIMA Parameters
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_pdq = None
    best_model = None

    for param in pdq:
        try:
            model_arima = ARIMA(y_reg_train_series, order=param)
            results_arima = model_arima.fit()
            if results_arima.aic < best_aic:
                best_aic = results_arima.aic
                best_pdq = param
                best_model = results_arima
        except:
            continue

    if best_model is not None:
        print(f"Best ARIMA model: ARIMA{best_pdq}")
        forecast = best_model.get_forecast(steps=len(series_hold_out))
        y_reg_pred_arima_diff = forecast.predicted_mean  # Predicted differenced values

        # Reverse differencing to match the original scale
        last_value = series_train["values"].iloc[-1]  # Last observed value in training data
        y_reg_pred_arima = np.cumsum(y_reg_pred_arima_diff) + last_value

        # Use the same index as the hold-out data
        y_reg_pred_arima.index = hold_out["date"]

        # Evaluate ARIMA Performance
        mae_arima = mean_absolute_error(series_hold_out["values"], y_reg_pred_arima)
        print(f"Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}")

        # Plot Hold-Out Data and Forecast
        plt.figure(figsize=(12, 6))
        plt.plot(
            series_train["date"],
            series_train["values"],
            label="Training Data",
            color="blue",
        )
        plt.plot(
            series_hold_out["date"],
            series_hold_out["values"],
            label="Hold-Out Data",
            color="green",
        )
        plt.plot(
            y_reg_pred_arima.index,
            y_reg_pred_arima.values,
            label="ARIMA Forecast",
            color="red",
        )
        plt.title("ARIMA Forecast vs Hold-Out Data")
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.legend()
        plt.savefig("arima_forecast_vs_holdout.png")
        plt.show()

        # Plot Residuals (ARIMA)
        residuals_arima = series_hold_out["values"].values - y_reg_pred_arima.values
        plt.figure(figsize=(12, 6))
        plt.plot(
            series_hold_out["date"],
            residuals_arima,
            label="ARIMA Residuals",
            color="purple",
        )
        plt.axhline(0, color="red", linestyle="--", label="Zero Error Line")
        plt.title("ARIMA Residuals")
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.legend()
        plt.savefig("arima_residuals.png")
        plt.show()
    else:
        print("No valid ARIMA model found.")


def suppress_warnings_2() -> None:
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    # Suppress warnings
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the ERCOT data
    df = pd.read_csv("ercot_load_data.csv")
    df["date"] = pd.to_datetime(df["date"])  # Ensure 'date' is in datetime format
    df["values"] = pd.to_numeric(
        df["values"], errors="coerce"
    )  # Convert 'values' to numeric
    df = df.sort_values("date")  # Sort by date
    df = df.dropna()  # Drop rows with missing or NaN values

    # Resample the data to hourly frequency
    df = (
        df.set_index("date").resample("h").mean().reset_index()
    )  # Resample and take the mean for each hour

    # Define hold-out period
    hold_out_hours = 24  # 24 hours = 1 day
    train = df.iloc[:-hold_out_hours]
    hold_out = df.iloc[-hold_out_hours:]

    # Generate Lagged Features
    df["lag_1"] = df["values"].shift(1)
    df["lag_2"] = df["values"].shift(2)
    df["rate_of_change"] = df["values"].diff()

    # Target for Classification: 1 if increase, 0 if decrease
    df["direction"] = (df["values"].shift(-1) > df["values"]).astype(int)

    # Target for Regression: Next Value
    df["next_value"] = df["values"].shift(-1)

    # Drop NaN Rows (after generating lag features)
    df = df.dropna()

    # Train-Test Split
    X = df[["values", "lag_1", "lag_2", "rate_of_change"]]
    y_class = df["direction"]
    y_reg = df["next_value"]
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)
    )

    # Classification Model: Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_class_train)

    # Add Classification Predictions as a Feature for Regression
    X_train_reg = X_train.copy()
    X_test_reg = X_test.copy()
    X_train_reg["direction_pred"] = clf.predict(X_train)
    X_test_reg["direction_pred"] = clf.predict(X_test)

    # Regression Model: Random Forest Regressor
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_reg, y_reg_train)

    # Predict Values
    y_class_pred = clf.predict(X_test)
    y_reg_pred = reg.predict(X_test_reg)

    # Evaluate Performance
    accuracy = accuracy_score(y_class_test, y_class_pred)
    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred)
    print(f"Classification Accuracy (Random Forest): {accuracy:.2f}")
    print(f"Regression Mean Absolute Error (Random Forest): {mae_rf:.2f}")

    # Correct Hold-Out Data Length
    X_hold_out = df[["values", "lag_1", "lag_2", "rate_of_change"]].iloc[
        -hold_out_hours:
    ]  # Features for hold-out period
    y_hold_out_actual = df["values"].iloc[
        -hold_out_hours:
    ]  # Actual values for hold-out period

    # Add classification predictions to hold-out features
    X_hold_out["direction_pred"] = clf.predict(X_hold_out)

    # Generate predictions for the hold-out period
    y_hold_out_pred = reg.predict(X_hold_out)

    # Combine Data for Plotting
    train_data = train[["date", "values"]]
    hold_out_data = hold_out[["date", "values"]].reset_index(drop=True)
    predicted_data = hold_out_data.copy()
    predicted_data["predicted_values"] = y_hold_out_pred

    # Plot Training Data, Hold-Out Data, and Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(train_data["date"], train_data["values"], label="Training Data", color="blue")
    plt.plot(
        hold_out_data["date"],
        hold_out_data["values"],
        label="Hold-Out Data (Actual)",
        color="green",
    )
    plt.plot(
        predicted_data["date"],
        predicted_data["predicted_values"],
        label="Random Forest Predictions",
        color="red",
    )
    plt.title("Random Forest Predictions vs Actual Values")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.savefig("rf_predictions_vs_actual.png")
    plt.show()


    # ARIMA Model
    y_reg_train_series = train["values"].diff().dropna()

    # Check Stationarity (Optional)
    adf_test = adfuller(y_reg_train_series)
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    if adf_test[1] > 0.05:
        print("Warning: Time series is not stationary!")

    # Grid Search for ARIMA Parameters
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_pdq = None
    best_model = None

    for param in pdq:
        try:
            model_arima = ARIMA(y_reg_train_series, order=param)
            results_arima = model_arima.fit()
            if results_arima.aic < best_aic:
                best_aic = results_arima.aic
                best_pdq = param
                best_model = results_arima
        except:
            continue

    if best_model is not None:
        print(f"Best ARIMA model: ARIMA{best_pdq}")
        forecast = best_model.get_forecast(steps=len(hold_out))
        y_reg_pred_arima_diff = forecast.predicted_mean  # Predicted differenced values

        # Reverse differencing to match the original scale
        last_value = train["values"].iloc[-1]  # Last observed value in training data
        y_reg_pred_arima = np.cumsum(y_reg_pred_arima_diff) + last_value

        # Evaluate ARIMA Performance
        mae_arima = mean_absolute_error(hold_out["values"], y_reg_pred_arima)
        print(f"Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}")

        # Plot Hold-Out Data and Forecast
        plt.figure(figsize=(12, 6))
        plt.plot(train["date"], train["values"], label="Training Data", color="blue")
        plt.plot(hold_out["date"], hold_out["values"], label="Hold-Out Data", color="green")
        plt.plot(
            hold_out["date"], y_reg_pred_arima.values, label="ARIMA Forecast", color="red"
        )
        plt.title("ARIMA Forecast vs Hold-Out Data")
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.legend()
        plt.savefig("arima_forecast_vs_holdout.png")
        plt.show()

        # Plot Residuals (ARIMA)
        residuals_arima = hold_out["values"].values - y_reg_pred_arima.values
        plt.figure(figsize=(12, 6))
        plt.plot(hold_out["date"], residuals_arima, label="ARIMA Residuals", color="purple")
        plt.axhline(0, color="red", linestyle="--", label="Zero Error Line")
        plt.title("ARIMA Residuals")
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.legend()
        plt.savefig("arima_residuals.png")
        plt.show()
    else:
        print("No valid ARIMA model found.")

    # Final Performance Comparison
    print(f"Classification Accuracy (Random Forest): {accuracy:.2f}")
    print(f"Regression Mean Absolute Error (Random Forest): {mae_rf:.2f}")
    if best_model is not None:
        print(f"Regression Mean Absolute Error (ARIMA): {mae_arima:.2f}")


def correct_hold_out_data_length() -> None:
    X_hold_out = df[['values', 'lag_1', 'lag_2', 'rate_of_change']].iloc[-hold_out_hours:]

    y_hold_out_actual = df['values'].iloc[-hold_out_hours:]

    X_hold_out['direction_pred'] = clf.predict(X_hold_out)

    y_hold_out_pred = reg.predict(X_hold_out)

    train_data = train[['date', 'values']]

    hold_out_data = hold_out[['date', 'values']].reset_index(drop=True)

    predicted_data = hold_out_data.copy()

    predicted_data['predicted_values'] = y_hold_out_pred

    plt.figure(figsize=(12, 6))

    plt.plot(train_data['date'], train_data['values'], label='Training Data', color='blue', linewidth=2)

    plt.plot(hold_out_data['date'], hold_out_data['values'], label='Hold-Out Data (Actual)', color='green', linewidth=2)

    plt.plot(predicted_data['date'], predicted_data['predicted_values'], label='Random Forest Predictions', color='red', linestyle='--', linewidth=2)

    plt.title('Random Forest Predictions vs Actual Values')

    plt.xlabel('Date')

    plt.ylabel('Values')

    plt.legend()

    plt.grid(True)

    plt.savefig('rf_predictions_vs_actual.png')

    plt.show()


def main() -> None:
    simulated_time_series_data()
    ensure_data_stationarity_optional_use_differenci()
    suppress_arima_warnings()
    suppress_arima_warnings_2()
    suppress_warnings()
    suppress_warnings_2()
    correct_hold_out_data_length()

if __name__ == "__main__":
    main()
