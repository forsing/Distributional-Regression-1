# https://medium.com/@guyko81/stop-predicting-numbers-start-predicting-distributions-0d4975db52ae
# https://github.com/guyko81/DistributionRegressor



"""
DistributionRegressor: Nonparametric Distributional Regression 
Lotto 7/39 predict probability  DistributionRegressor
""" 

import numpy as np
import pandas as pd
from distribution_regressor import DistributionRegressor
import matplotlib.pyplot as plt

CSV_PATH = "/data/loto7hh_4592_k27.csv"
COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
SEED = 39
np.random.seed(SEED)


def load_draws(path):
    df = pd.read_csv(path)
    if all(c in df.columns for c in COLS):
        return df[COLS].values.astype(float)
    return pd.read_csv(path, header=None).iloc[:, :7].values.astype(float)


def enforce_loto_7_39(nums):
    nums = np.rint(np.asarray(nums, dtype=float)).astype(int)
    mins = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
    maxs = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)
    nums = np.clip(nums, mins, maxs)
    nums = np.sort(nums)
    for i in range(7):
        low = mins[i] if i == 0 else max(mins[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)
    for i in range(6, -1, -1):
        high = maxs[i] if i == 6 else min(maxs[i], nums[i + 1] - 1)
        nums[i] = min(nums[i], high)
    for i in range(7):
        low = mins[i] if i == 0 else max(mins[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)
    return nums


draws = load_draws(CSV_PATH)
X_train = draws[:-1]
Y_train = draws[1:]
X_next = draws[-1:].copy()

# 1. Initialize + 2. Train (po pozicijama 1..7)
pred_mode = []
pred_median = []
models = []

for pos in range(7):
    y_train = Y_train[:, pos]
    model = DistributionRegressor(
        n_bins=80,              # Number of grid points (higher = more resolution, more RAM)
                                # Resolution of the distribution grid
        n_estimators=320,       # Number of boosting trees
        use_base_model=False,   # If True, learns residual CDF around a base LGBM prediction
        monte_carlo_training=True,  # If True, sample grid points instead of full expansion
        mc_samples=7,           # MC sample points per observation (when MC enabled)
        mc_resample_freq=100,   # Resample grid points every N trees (lower = better coverage)
        learning_rate=0.06,     # Learning rate
        random_state=SEED,      # Seed
    )
    model.fit(X_train, y_train)
    models.append(model)

    # 3. Predict Points
    y_mode = float(np.asarray(model.predict_mode(X_next)).ravel()[0])        # Mode (Most likely value / Peak)
    y_median = float(np.asarray(model.predict_quantile(X_next, 0.5)).ravel()[0])
    pred_mode.append(y_mode)
    pred_median.append(y_median)

# Ispis 2 tražene predikcije
y_pred = enforce_loto_7_39(pred_mode)
Prediction = enforce_loto_7_39(pred_median)
print("y_pred (mode):", y_pred)
print("Prediction (median):", Prediction)
"""
y_pred (mode): [ 1  5 x y z 29 37]
Prediction (median): [ 3  8 x y z 30 37]
"""



# 4. Predict Intervals & Uncertainty (prikaz samo za poziciju 1)
# 10th and 90th percentiles (80% confidence interval)
lower = models[0].predict_quantile(X_next, 0.1)
upper = models[0].predict_quantile(X_next, 0.9)
print("pos1 interval [10%,90%]:", float(lower[0]), float(upper[0]))

# 5. Predict Full Distribution (prikaz samo za poziciju 1)
grids, dists, offsets = models[0].predict_distribution(X_next)
# grids: (n_samples, n_bins) - Per-sample grid points
# dists: (n_samples, n_bins) - Probability mass for each sample

# Predict distribution for a single sample
plt.plot(grids[0], dists[0], label="Predicted PMF (pos1)")
plt.axvline(y_pred[0], color="r", linestyle="--", label="Predicted mode")
plt.legend()
plt.show()

