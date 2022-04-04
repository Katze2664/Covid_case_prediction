import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit


def casesDateSourceCount(path):
    df = pd.read_csv(path)
    df["count"] = 1

    df2 = df.groupby(["diagnosis_date", "acquired"]).count()

    df3 = df2.unstack(level=-1).fillna(0).astype(int)
    df3.columns = df3.columns.droplevel()
    df3["Non-travel"] = df3["Acquired in Australia, unknown source"] + df3["Contact with a confirmed case"] + df3["Under investigation"]
    df3["Total"] = df3["Acquired in Australia, unknown source"] + df3["Contact with a confirmed case"] + df3["Under investigation"] + df3["Travel overseas"]

    df4 = df3.loc["2021-07-29":]
    df4.index = df4.index.map(lambda s: (pd.Timestamp(s) - pd.Timestamp("2021-07-29")) // pd.Timedelta("1d"))
    df4.index.name = "Days since 29th July 2021"
    df4.columns.name = None

    return df4


def exp_func(x, a, b, c):
    y = a * np.exp(b * x) + c
    return y

class Cases:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def fit_exp(self, guess = [1, 1, 1]):
        fit = curve_fit(exp_func, self.xdata, self.ydata, guess)
        self.exp_a = fit[0][0]
        self.exp_b = fit[0][1]
        self.exp_c = fit[0][2]
        self.exp_fit_data = exp_func(self.xdata, self.exp_a, self.exp_b, self.exp_c)

def rSquared(ydata, fit_data):
    fit_residuals = (ydata - fit_data)**2
    fit_res_sum = np.sum(fit_residuals)

    mean = np.mean(ydata)
    mean_residuals = (ydata - mean)**2
    mean_res_sum = np.sum(mean_residuals)

    r_squared = 1 - (fit_res_sum / mean_res_sum)
    return r_squared

df_old = casesDateSourceCount("NCOV_COVID_Cases_by_Source_20210906.csv")
df_new = casesDateSourceCount("NCOV_COVID_Cases_by_Source_20210917.csv")

old = Cases(np.array(df_old.index), np.array(df_old["Non-travel"]))
new = Cases(np.array(df_new.index), np.array(df_new["Non-travel"]))

old.fit_exp()
old.exp_r2 = rSquared(old.ydata, old.exp_fit_data)
print("The green line is an exponential fit for the green data points.")
print(f"R-squared over green data points = {round(old.exp_r2, 2)}")

old_fit_extrapolated = exp_func(new.xdata, old.exp_a, old.exp_b, old.exp_c)
old_fit_extrapolated_r2 = rSquared(new.ydata, old_fit_extrapolated)
print("\nThe green exponential fit does not do well at predicting the new red data points.")
print(f"R-squared over all data points = {round(old_fit_extrapolated_r2, 2)}")


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(old.xdata, old.ydata, "go",
         old.xdata, old.exp_fit_data, "g-")
ax1.set_ylim([0, 900])
ax1.set_xlim([0, 50])
ax1.set_xlabel("Days since 29th July")
ax1.set_ylabel("Victorian non-travel new covid cases per day")
ax1.set_title("Covid data up to 6th September 2021")

ax2.plot(old.xdata, old.ydata, "go",
         new.xdata[39:], new.ydata[39:], "ro",
         new.xdata, old_fit_extrapolated, "g-")
ax2.set_ylim([0, 900])
ax2.set_xlim([0, 50])
ax2.set_xlabel("Days since 29th July")
ax1.set_ylabel("Victorian non-travel new covid cases per day")
ax2.set_title("Covid data up to 17th September 2021")

plt.show()
