import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

def plot_ac(series, diff=None):
    if diff:
        series = series.diff(periods=diff).iloc[1:]
    plt.title(f"Series (diff{diff})")
    series.plot()
    plt.title(f"ACF (diff{diff})")
    plot_acf(series)
    plt.title(f"PACF (diff{diff})")
    plot_pacf(series)
    plt.show()
    
