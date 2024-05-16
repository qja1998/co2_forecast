import numpy as np

def get_data(df, scaling=True):
    time        = np.arange(0, 400, 0.1)
    #amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
    scaler = None

    if scaling:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler() 
        amplitude = scaler.fit_transform(df.to_numpy().reshape(-1, 1)).reshape(-1)
    else:
        amplitude = df.to_numpy()
    
    sampels = int(len(amplitude) * 0.8)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    return train_data, test_data, scaler