import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def plot_slect(series, t, data, acf, pacf, lags, d=None):
    sub_plot_size = sum([data, acf, pacf])
    _, ax = plt.subplots(sub_plot_size, 1, figsize=(15, sub_plot_size * 4))
    
    pos = 0
    if d is not None:
        series_tmp = series.diff(periods=d).dropna()
    else:
        series_tmp = series.copy()

    if data:
        series_tmp.plot(ax=ax[pos])
        pos += 1
    if acf:
        plot_acf(series_tmp, ax=ax[pos], lags=lags)
        pos += 1
    if pacf:
        plot_pacf(series_tmp, ax=ax[pos], lags=lags)
        pos += 1
        
    plt.title(f"[{t}] diff : {d}")
    
    
def plot_ac(series, t, diff=[], data=True, acf=True, pacf=True, lags=1):
    
    plot_slect(series, t, data, acf, pacf, lags)

    for d in diff:
        plot_slect(series, t, data, acf, pacf, lags, d)

    plt.show()
    
