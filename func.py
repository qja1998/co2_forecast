# https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-multistep.py

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

# from matplotlib import pyplot


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]


def create_input_sequences(input_data, tw, output_window=1):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_pred_input_data(data_array, input_window, output_window=1):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(data_array.reshape(-1, 1)).reshape(-1)

    sequence = create_input_sequences(amplitude,input_window)
    sequence = sequence[:-output_window] #todo: fix hack?

    return sequence, scaler

def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


def predict_future(eval_model, data_source_list, steps, input_window, output_window=1):
    eval_model.eval()
    
    for data_source, type, scaler in data_source_list:
        _ , data = get_batch(data_source, 0,1)
        with torch.no_grad():
            for i in range(0, steps,1):
                input = torch.clone(data[-input_window:])
                input[-output_window:] = 0     
                output = eval_model(data[-input_window:])                        
                data = torch.cat((data, output[-1:]))
                
        data = data.cpu().view(-1)
        data = scaler.inverse_transform(data.reshape(1, -1)).reshape(-1, 1)
    return data


def get_prediction(data, model, device, pred_len=180):

    data = np.array(data)
    data, scaler = get_pred_input_data(data)
    data.to(device)
    data_list = [(data, '', scaler)]
    pred = predict_future(model, data_list, pred_len)
    
    return pred