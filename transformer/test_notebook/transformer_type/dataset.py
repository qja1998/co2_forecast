import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer

scaler_dict = {
    'mm' : MinMaxScaler(),
    'sd' : StandardScaler(),
    'rb' : RobustScaler(),
    # 'nr' : Normalizer(),
}
SEP = -100

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]

def _create_input_sequences(input_data, tw, output_window, diff, mean_std):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        intput_seq = input_data[i:i+tw][:-output_window]
        train_seq = intput_seq
        if diff:
            diff_seq = np.append(np.diff(intput_seq), np.array([SEP]))
            train_seq = np.append(diff_seq, train_seq)
        
        if mean_std:
            mean_std_val = np.array([np.mean(intput_seq), SEP, np.std(intput_seq)])
            mean_std_val = np.append(mean_std_val, np.array([SEP]))
            train_seq = np.append(mean_std_val, train_seq)

        #train_label = input_data[i+output_window:i+tw+output_window]
        train_label = np.append(train_seq, input_data[i:i+tw][-output_window:])
        train_seq = np.append(train_seq , output_window * [0])

        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_data(df, input_window, output_window, scaler_name, diff=False, mean_std=False):
    time        = np.arange(0, 400, 0.1)
    #amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    df = df.to_numpy().reshape(-1, 1)
    scaler = scaler_dict[scaler_name]
    amplitude = scaler.fit_transform(df).reshape(-1)
    
    sampels = int(len(amplitude) * 0.8)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = _create_input_sequences(train_data, input_window + output_window, output_window, diff, mean_std)
    # train_sequence = train_sequence[:-output_window] #todo: fix hack?

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = _create_input_sequences(test_data, input_window + output_window, output_window, diff, mean_std)
    # test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence, test_data, scaler

def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target