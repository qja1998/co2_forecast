import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

def plot_ac(series, t, diff=[]):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))


    series.plot(ax=ax[0])
    plot_acf(series, ax=ax[1])
    plot_pacf(series, ax=ax[2])

    plt.title(f"[{t}] original")

    for d in diff:
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        series_diff = series.diff(periods=d).iloc[1:]
        
        series_diff.plot(ax=ax[0])
        plot_acf(series_diff, ax=ax[1])
        plot_pacf(series_diff, ax=ax[2])

        plt.title(f"[{t}] diff : {d}")
    
    plt.show()
    
