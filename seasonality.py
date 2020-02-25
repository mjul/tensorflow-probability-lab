"""Experiments with seasonality"""

# Based on the tutorial
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns

# For simplicity we use whole months
num_months = 12

from_month = np.datetime64('2020-01', 'M')  # included
to_month = from_month + np.timedelta64(num_months, 'M')  # not included
dates = np.arange(from_month, to_month, dtype='datetime64[D]')
num_days = (to_month - from_month).astype('timedelta64[D]').astype(np.int32)

# Create some observations on these days
first_in_month = dates.astype('datetime64[M]')
day_of_month = np.ndarray.astype(dates - first_in_month + 1, dtype=np.int32)
observations = day_of_month + 5 * np.random.standard_normal(dates.size)

# Let's have a look
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(dates, observations, lw=2, label="Observations")
fig.autofmt_xdate()
fig.show()

# here a season is a day of the month
# some months have fewer than 31 days, so we need to adjust for that
# to make seasonality on day-of-month work correctly

day_of_month_num_seasons = 31
day_of_month_steps_per_season_flat = np.zeros(num_months * day_of_month_num_seasons, dtype=np.int32)

days_in_month = (first_in_month + np.timedelta64(1, dtype='M')).astype('datetime64[D]') - first_in_month

for m in range(num_months):
    month_first_idx = m * day_of_month_num_seasons
    month_last_idx = month_first_idx + days_in_month[month_first_idx].astype(np.int32)
    # All days of the month have 1 step (one day)
    # In shorter months the high-numbered days of the month do not exist so the step is 0 (set as default above)
    day_of_month_steps_per_season_flat[month_first_idx:month_last_idx] = 1

day_of_month_steps_per_season = day_of_month_steps_per_season_flat.reshape(-1, day_of_month_num_seasons)


def build_model(observed_time_series):
    seasonal_day_of_month = tfp.sts.Seasonal(
        num_seasons=31,
        num_steps_per_season=day_of_month_steps_per_season,
        observed_time_series=observed_time_series,
        name='day_of_month')
    model = tfp.sts.Sum([seasonal_day_of_month], observed_time_series=observed_time_series)
    return model


# Training data and a hold-out test set
num_training_months = num_months - 1
num_training_days = (
        (from_month + np.timedelta64(num_training_months, 'M')).astype('datetime64[D]') - from_month).astype(
    np.int32)
training_dates = dates[:num_training_days]
training_observations = observations[:num_training_days]

num_test_days = num_days - num_training_days
test_dates = dates[num_training_days:]
test_observations = observations[num_training_days:]

model = build_model(training_observations)

variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

num_variational_steps = 200

optimizer = tf.optimizers.Adam(learning_rate=.1)


# Using fit_surrogate_posterior to build and optimize the variational loss function.
# Compilation not yet available since we are using Windows
@tf.function(experimental_compile=False)
def train():
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_log_prob(observed_time_series=training_observations),
        surrogate_posterior=variational_posteriors,
        optimizer=optimizer,
        num_steps=num_variational_steps)
    return elbo_loss_curve


elbo_loss_curve = train()

plt.plot(elbo_loss_curve)
plt.suptitle("Evidence Lower Bound (ELBO) Loss")
plt.show()

# Draw samples from the variational posterior.
q_samples = variational_posteriors.sample(50)

print("Inferred parameters:")
for param in model.parameters:
    print("{}: {} +- {}".format(param.name,
                                np.mean(q_samples[param.name], axis=0),
                                np.std(q_samples[param.name], axis=0)))

num_forecast_steps = num_test_days
forecast_dist = tfp.sts.forecast(
    model,
    observed_time_series=training_observations,
    parameter_samples=q_samples,
    num_steps_forecast=num_forecast_steps)

num_samples = 10

forecast_mean, forecast_scale, forecast_samples = (
    forecast_dist.mean().numpy()[..., 0],
    forecast_dist.stddev().numpy()[..., 0],
    forecast_dist.sample(num_samples).numpy()[..., 0])


def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
    """Plot a forecast distribution against the 'true' time series."""
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    num_steps = len(y)
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast

    ax.plot(x, y, lw=2, color=c1, label='ground truth')

    forecast_steps = np.arange(
        x[num_steps_train],
        x[num_steps_train] + num_steps_forecast,
        dtype=x.dtype)

    ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

    ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
            label='forecast')
    ax.fill_between(forecast_steps,
                    forecast_mean - 2 * forecast_scale,
                    forecast_mean + 2 * forecast_scale, color=c2, alpha=0.2)

    ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
    yrange = ymax - ymin
    ax.set_ylim([ymin - yrange * 0.1, ymax + yrange * 0.1])
    ax.set_title("{}".format(title))
    ax.legend()

    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()

    return fig, ax


fig, ax = plot_forecast(
    dates, observations,
    forecast_mean, forecast_scale, forecast_samples,
    # £x_locator=co2_loc,
    # £x_formatter=co2_fmt,
    title="Observations forecast")
ax.axvline(dates[-num_forecast_steps], linestyle="--")
ax.legend(loc="upper left")
ax.set_ylabel("Value")
ax.set_xlabel("Date")
fig.autofmt_xdate()

fig.show()
