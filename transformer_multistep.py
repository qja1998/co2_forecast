import sys, os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import time
import math
import random
from matplotlib import pyplot
import wandb

# from pytorch_forecasting.metrics.point import SMAPE

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device('cpu')

# This concept is also called teacher forceing. 
# The flag decides if the loss will be calculted over all 
# or just the predicted values.
calculate_loss_over_all_values = False

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
#print(out)

# Set parameters
input_window = 180
output_window = 1
batch_size = 32 # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_loss = float("inf")
epochs = 300 # The number of epochs
log_epoch = 50
best_model = None
pred_step = 180

feature_size=1024
num_layers=1
lr = 1e-2

RESULT_PATH = f"./transformer_results/single_model/{input_window}-{output_window}_{batch_size}_{feature_size}-{num_layers}_{lr}_{epochs}"
RESULT_TXT_PATH = RESULT_PATH + "/output.txt"

if not os.path.isdir(RESULT_PATH):
    os.mkdir(RESULT_PATH)
if not os.path.isdir(RESULT_PATH + f"/future{pred_step}"):
    os.mkdir(RESULT_PATH + f"/future{pred_step}")
with open(RESULT_TXT_PATH, 'w') as f:
    f.write('')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
       

class TransAm(nn.Module):
    def __init__(self,feature_size=256,num_layers=3,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_input_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        #train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_data(type='A'):
    time        = np.arange(0, 400, 0.1)
    #amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
    from pandas import read_csv
    series = read_csv('data/korea/kor_gas_day.csv', header=0, index_col=0)
    series = series.loc[series.type == type]
    series = series.drop(['type','year','month','day'], axis=1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    #amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    
    
    sampels = int(len(amplitude) * 0.8)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_input_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_input_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device), test_data.to(device), scaler

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


def train(train_data):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            with open(RESULT_TXT_PATH, 'a') as f:
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}\n'.format(
                        epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source_list, epoch, scaler):
    eval_model.eval() 
    total_loss = 0
    mse = nn.MSELoss()
    total_mse = 0
    mae = nn.L1Loss()
    total_mae = 0
    # smape = SMAPE()
    total_smape = 0

    true_val = []
    predictions = []
    with torch.no_grad():
        data, _ = get_batch(val_data, 0, 1)
        for data_source, type, scaler in data_source_list:
            for i in range(len(val_data) - output_window):
                _, true_y = get_batch(val_data, i, 1)
                output = eval_model(data)
                tmp = data[:-1]
                data = torch.cat([tmp, output[:1]], dim=0)

            total_mse += mse(output, true_y.to(device))
            total_mae += mae(output, true_y.to(device))
            true_val.append(float(scaler.inverse_transform(true_y[0].detach().cpu().numpy())))
            predictions.append(float(scaler.inverse_transform(output[0].detach().cpu().numpy().squeeze().reshape(-1, 1))))
            
            
    with open(RESULT_TXT_PATH, 'a') as f:
        f.write(f'\nMSE: {total_mse / len(val_data)}, MAE: {total_mae / len(val_data)}\n')
    x = np.arange(len(true_val))
        
    pyplot.figure(figsize=(20, 15))
    pyplot.rc('font', size=20) 
    pyplot.subplot(413)
    pyplot.plot(x[:180], true_val[:180], label='true', c='blue')
    pyplot.plot(x[:180], predictions[:180], label='predictions', c='red')
    pyplot.legend()
    pyplot.savefig(RESULT_PATH + '/transformer-epoch%d.png'%epoch)
    pyplot.close()
    
    return total_loss / i, total_mse / i, total_mae / i, total_smape / i


def predict_future(eval_model, data_source_list, steps):
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    total_mse = 0
    total_mae = 0

    eval_model.eval()
    pyplot.figure(figsize=(20, 10))
    sub_num = 421
    for data_source, type, scaler in data_source_list:

        _ , data = get_batch(data_source, 0,1)

        with torch.no_grad():
            for i in range(0, steps,1):
                input = torch.clone(data[-input_window:])
                input[-output_window:] = 0     
                output = eval_model(data[-input_window:])                        
                data = torch.cat((data, output[-1:]))
                
        data = data.cpu().view(-1)
        true_future = data_source[input_window][1].cpu().view(-1)
        true_val = list(data[:input_window]) + list(true_future)
        
        data_tensor = torch.Tensor(data[input_window:])
        data_true_future = torch.Tensor(true_future)
        mse_score = mse(data_tensor, data_true_future.squeeze())
        mae_score = mae(data_tensor, data_true_future.squeeze())

        total_mse += mae_score
        total_mae += mae_score

        data = scaler.inverse_transform(data.reshape(-1, 1))
        true_val = scaler.inverse_transform(np.array(true_val).reshape(-1, 1))

        with open(RESULT_TXT_PATH, 'a') as f:
            f.write(f"{type} {epoch} epochs - mse: {mse_score}, mae: {mae_score}")
        wandb.log({'epoch': epoch, 'mse_score': mse_score, 'mae_score': mae_score})
        pyplot.subplot(sub_num)
        pyplot.plot(true_val,color="red", label="true")
        pyplot.plot(range(input_window, input_window + steps), data[input_window:],color="blue", label='predictions')
        pyplot.grid(True, which='both')
        pyplot.title(type)
        pyplot.legend()
        sub_num += 1

    num_types = len(data_source_list)
    with open(RESULT_TXT_PATH, 'a') as f:
        f.write(f"Total {epoch} epochs - mse: {total_mse / num_types}, mae: {total_mae / num_types}\n")

    pyplot.savefig(RESULT_PATH + '/future%d/transformer-future%d.png'%(steps, epoch))
    pyplot.close()
        
# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
# auch zu denen der predict_future
def evaluate(eval_model, data_source_list):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    total_val_len = 0
    with torch.no_grad():
        for data_source, _, _ in data_source_list:
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = get_batch(data_source, i,eval_batch_size)
                output = eval_model(data)
                if calculate_loss_over_all_values:
                    total_loss += len(data[0])* criterion(output, targets).cpu().item()
                else:                                
                    total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()
            total_val_len += len(data_source)
    return total_loss / total_val_len


wandb.init(
        # set the wandb project where this run will be logged
        project="co2 emission forecasting",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epochs,
        }
    )
wandb.run.name = RESULT_PATH.split('/')[-1]
wandb.run.save()

# make dataset

train_data, val_data, scaler0 = get_data('I')
train_data1, val_data1, scaler1 = get_data('A')
train_data2, val_data2, scaler2 = get_data('B')
train_data3, val_data3, scaler3 = get_data('C')
train_data4, val_data4, scaler4 = get_data('D')
train_data5, val_data5, scaler5 = get_data('E')
train_data6, val_data6, scaler6 = get_data('G')
train_data7, val_data7, scaler7 = get_data('H')

train_data_list = ((train_data, 'I', scaler0),
                (train_data1, 'A', scaler1),
                (train_data2, 'B', scaler2),
                (train_data3, 'C', scaler3),
                (train_data4, 'D', scaler4),
                (train_data5, 'E', scaler5),
                (train_data6, 'G', scaler6),
                (train_data7, 'H', scaler7)
                )

val_data_list = ((val_data, 'I', scaler0),
                (val_data1, 'A', scaler1),
                (val_data2, 'B', scaler2),
                (val_data3, 'C', scaler3),
                (val_data4, 'D', scaler4),
                (val_data5, 'E', scaler5),
                (val_data6, 'G', scaler6),
                (val_data7, 'H', scaler7)
                )

# make model
model = TransAm(feature_size=feature_size, num_layers=num_layers).to(device)

criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

for epoch in tqdm(range(1, epochs + 1)):
    epoch_start_time = time.time()
    for train_data, type, scaler in train_data_list:
        train(train_data)
    
    if(epoch % log_epoch == 0):
        # val_loss, mse_loss, mae_loss, smape_loss = plot_and_loss(model, val_data, epoch, scaler0)
        predict_future(model, val_data_list, pred_step)
    else:
        val_loss = evaluate(model, val_data_list)
        
    with open(RESULT_TXT_PATH, 'a') as f:
        f.write('-' * 89 + '\n')
        f.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss(mse) {:5.5f} | valid ppl {:8.2f}\n'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        f.write('-' * 89 + '\n')
    wandb.log({'epoch': epoch, 'val_loss': val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step() 

torch.save(model.state_dict(), f"{RESULT_PATH}/{input_window}_{batch_size}_{feature_size}-{num_layers}_{lr}_{epochs}-best_model-{best_val_loss:.5f}.pt")

#src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number) 
#out = model(src)
#
#print(out)
#print(out.shape)
