import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def plot_slect(series, t, data, acf, pacf, d=None):
    sub_plot_size = sum([data, acf, pacf])
    fig, ax = plt.subplots(sub_plot_size, 1, figsize=(10, sub_plot_size * 4))
    pos = 0
    if d is not None:
        series = series.diff(periods=d).iloc[1:]
    if data:
        series.plot(ax=ax[pos])
        pos += 1
    if acf:
        plot_acf(series, ax=ax[pos])
        pos += 1
    if pacf:
        plot_pacf(series, ax=ax[pos])
        pos += 1
        
    plt.title(f"[{t}] diff : {d}")
    
    
def plot_ac(series, t, diff=[], data=True, acf=True, pacf=True):
    
    plot_slect(series, t, data, acf, pacf)

    for d in diff:
        plot_slect(series, t, data, acf, pacf, d)

    plt.show()
    
