import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import wandb

from .dataset import get_batch

from torchmetrics.regression import SymmetricMeanAbsolutePercentageError as SMAPE

calculate_loss_over_all_values = False


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
    def __init__(self,feature_size=256,num_layers=3,d_ff=2048,dropout=0.1, output_size=1):
        super(TransAm, self).__init__()

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, output_size)
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

# class TransAm(nn.Module):
#     def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout):
#         super().__init__()
 
#         # INFO
#         self.model_type = "Transformer"
#         self.dim_model = dim_model
 
#         # LAYERS
#         self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout, max_len=5000)
#         self.embedding = nn.Embedding(num_tokens, dim_model)
#         self.transformer = nn.Transformer(
#             d_model=dim_model,
#             nhead=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dropout=dropout,
#         )
#         self.out = nn.Linear(dim_model, num_tokens)
 
#     def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
#         # src, Tgt size -> (batch_size, src sequence length)
 
#         # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
#         src = self.embedding(src) * math.sqrt(self.dim_model)
#         tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
#         src = self.positional_encoder(src)
#         tgt = self.positional_encoder(tgt)
 
#         src = src.permute(1,0,2)
#         tgt = tgt.permute(1,0,2)
 
#         # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
#         transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
#         out = self.out(transformer_out)
 
#         return out
 
#     def get_tgt_mask(self, size) -> torch.tensor:
#         mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
#         mask = mask.float()
#         mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
#         mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
 
#         return mask
 
#     def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
#         return (matrix == pad_token)

def train(model, train_data, type, epoch, optimizer, scheduler, criterion, batch_size, input_window, output_window, RESULT_TXT_PATH):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size, input_window + output_window)
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
                f.write('{} | epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}\n'.format(
                        type, epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        
    return model, optimizer, scheduler, loss.item()


def predict_future(eval_models, data_source_list, epoch, steps, input_window, output_window, RESULT_TXT_PATH, RESULT_PATH):
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    smape = SMAPE()

    total_mse = 0
    total_mae = 0
    total_smape = 0

    origin_step = steps
    
    pyplot.figure(figsize=(20, 10))
    sub_num = 1
    for data_source, type, scaler in data_source_list:
        steps = origin_step

        eval_model, _, _ = eval_models[type]
        eval_model.eval()

        _ , data = get_batch(data_source, 0, 1, input_window + output_window)

        with torch.no_grad():
            if steps == 1:
                data = eval_model(data[-input_window:])
                steps = output_window
                
            else:
                for i in range(0, steps):
                    output = eval_model(data[-input_window:])
                    data = torch.cat((data[:-1], output[-1:], torch.Tensor([[[.0]]]).to(data.device)))
                
        
        true_val_len = steps + input_window
        data = data.cpu().view(-1)
        true_future = np.array([data_source[i][1][0].cpu() for i in range(true_val_len)])
        true_val = np.array([data_source[i][1][0].cpu() for i in range(true_val_len)])
        
        data_tensor = torch.Tensor(data[-steps:])
        data_true_future = torch.Tensor(true_future)
        mse_score = mse(data_tensor, data_true_future[-steps:].squeeze())
        mae_score = mae(data_tensor, data_true_future[-steps:].squeeze())
        smape_score = smape(data_tensor, data_true_future[-steps:].squeeze())

        total_mse += mse_score
        total_mae += mae_score
        total_smape += smape_score

        data = scaler.inverse_transform(data.reshape(-1, 1))
        true_val = scaler.inverse_transform(np.array(true_val).reshape(-1, 1))

        with open(RESULT_TXT_PATH, 'a') as f:
            f.write(f"{type} {epoch} epochs - mse: {mse_score}, mae: {mae_score}\n")
        pyplot.subplot(5, 2, sub_num)
        pyplot.plot(true_val,color="red", label="true")
        pyplot.plot(range(input_window, input_window + steps), data[-steps-1:-1],color="blue", label='predictions')
        pyplot.grid(True, which='both')
        pyplot.title(type)
        pyplot.legend()

        wandb.log({f"pred_{type}": data_tensor, f"true_{type}": data_true_future[-steps:]})

        sub_num += 1

    num_types = len(data_source_list)
    mse_result, mae_result, smape_result = total_mse / num_types, total_mae / num_types, smape_score / num_types
    with open(RESULT_TXT_PATH, 'a') as f:
        f.write(f"Total {epoch} epochs - mse: {mse_result}, mae: {mae_result}, sampe: {smape_result}\n")
        wandb.log({"pred_MSE": mse_result, "pred_MAE": mae_result, 'pred_SMAPE' : smape_result})

    pyplot.savefig(RESULT_PATH + '/future%d/transformer-future%d.png'%(steps, epoch))
    pyplot.close()
    return mse_result, mae_result, smape_result
        
# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
# auch zu denen der predict_future
def evaluate(eval_model, data_source, criterion, input_window, output_window):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    total_val_len = 0
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size, input_window + output_window)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()
        total_val_len += len(data_source)
    return total_loss / total_val_len