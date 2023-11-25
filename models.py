import torch.nn as nn

class TensorExtractor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor
class ReshapeData(nn.Module):
    def __init__(self, seq_len, is_input=True):
        super(ReshapeData, self).__init__()
        self.seq_len = seq_len
        self.is_input = is_input
    def forward(self, x):
        if self.is_input:
            return x.unsqueeze(1)
        else:
            return x.squeeze(1)
    
class Model(nn.Module):
    def __init__(self, model_name, input_size, hidden_size, num_layer, conv=False, conv_size=1, kernel_size=2, stride=1, padding=0):
        super(Model, self).__init__()
        model_layers = []
        model_layers.append(nn.BatchNorm1d(input_size))
        if conv:
            conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
            model_layers.append(ReshapeData(input_size, is_input=True))
            model_layers.append(conv)
            model_layers.append(ReshapeData(input_size, is_input=False))
            input_size = input_size - 1
        model_layers.append(nn.BatchNorm1d(input_size))
        if model_name == 'lstm':
            model_layers.append(nn.LSTM(input_size, hidden_size, num_layer))
        elif model_name == 'gru':
            model_layers.append(nn.GRU(input_size, hidden_size, num_layer))
        elif model_name == 'rnn':
            model_layers.append(nn.RNN(input_size, hidden_size, num_layer))
        model_layers.append(TensorExtractor())
        model_layers.append(nn.BatchNorm1d(hidden_size))
        model_layers.append(nn.Linear(hidden_size, 1))
        model_layers.append(nn.ReLU())
        self.model = nn.Sequential(*model_layers)
    
    def forward(self, x):
        return self.model(x)