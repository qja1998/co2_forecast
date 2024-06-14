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
    def __init__(self, iw, ow, feature_size=256,num_layers=3,d_ff=2048,dropout=0.1, output_size=1):
        super(TransAm, self).__init__()

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, output_size)

        self.linear =  nn.Sequential(
            nn.Linear(feature_size, feature_size//2),
            nn.ReLU(),
            nn.Linear(feature_size//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw+ow-1, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        )

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
        # output = self.decoder(output)
        output = self.linear(output.transpose(0,1))[:,:,0]
        output = self.linear2(output).transpose(0, 1)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerFull(nn.Module):
    # Constructor
    def __init__( self, num_tokens=60, dim_model=256, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1, ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(d_model=dim_model, max_len=5000)
        # self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src, Tgt size 는 반드시 (batch_size, src sequence length) 여야 합니다.

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # src = self.embedding(src) * math.sqrt(self.dim_model)
        # tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)


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
            loss = criterion(output[-output_window:], targets[-output_window:].squeeze())
    
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

SEP = -100
def predict_future(eval_models, data_source_list, epoch, steps, input_window, output_window, RESULT_TXT_PATH, RESULT_PATH, diff, mean_std, wandb_log=False):
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    smape = SMAPE()

    # total_mse = 0
    # total_mae = 0
    # total_smape = 0

    origin_step = steps
    
    pyplot.figure(figsize=(20, 10))
    sub_num = 1
    total_pred, total_true = np.array([]), np.array([])
    for data_source, type, scaler in data_source_list:
        scaler, diff_scaler = scaler
        steps = origin_step

        eval_model, _, _ = eval_models[type]
        eval_model.eval()

        data, _ = get_batch(data_source, 0, 1, input_window + output_window)

        with torch.no_grad():
            if steps == 1:
                pred_output = eval_model(data)
                steps = output_window
                pred_output = pred_output.squeeze().squeeze().cpu()[-steps:]

            else:
                if diff or mean_std:
                    split_id = 0
                    data_list = list(data.squeeze().squeeze().cpu())
                    for i, d in enumerate(data_list[::-1]):
                        if d == SEP:
                            split_id = len(data_list) - i
                            break
                pred_output = []

                for _ in range(0, steps):
                    output = eval_model(data)
                    pred_output.append(output[-1].squeeze().squeeze().cpu())
                    data = data.squeeze().squeeze()
                    next_seq = np.append(np.array(data[split_id + 1:-2].cpu()), np.array(output[-1].cpu()))
                    pred_seq = next_seq.copy()
                    if diff:
                        diff_seq = np.append(np.diff(next_seq), np.array([SEP]))
                        pred_seq = np.append(diff_seq, pred_seq)
                    if mean_std:
                        mean_std_val = np.array([np.mean(next_seq), SEP, np.std(next_seq)])
                        mean_std_val = np.append(mean_std_val, np.array([SEP]))
                        pred_seq = np.append(mean_std_val, pred_seq)
                    data = torch.Tensor(np.append(pred_seq, np.array(.0))).to(data.device).unsqueeze(-1).unsqueeze(-1)
        
        true_val_len = steps + input_window
        true_future = np.array([data_source[i][1][0].cpu() for i in range(true_val_len)])
        true_val = np.array([data_source[i][1][0].cpu() for i in range(true_val_len)])
        
        data_tensor = torch.Tensor(pred_output)
        data_true_future = torch.Tensor(true_future)
        mse_score = mse(data_tensor, data_true_future[-steps:].squeeze())
        mae_score = mae(data_tensor, data_true_future[-steps:].squeeze())
        smape_score = smape(data_tensor, data_true_future[-steps:].squeeze())

        # total_mse += mse_score
        # total_mae += mae_score
        # total_smape += smape_score

        total_pred += list(pred_output)
        total_true += list(true_future)

        pred_output = scaler.inverse_transform(np.array(pred_output).reshape(-1, 1))
        true_val = scaler.inverse_transform(np.array(true_val).reshape(-1, 1))

        with open(RESULT_TXT_PATH, 'a') as f:
            f.write(f"{type} {epoch} epochs - MSE: {mse_score}, MAE: {mae_score}, SAMPE: {smape_score}\n")
        pyplot.subplot(5, 2, sub_num)
        pyplot.plot(true_val,color="red", label="true")
        pyplot.plot(range(input_window, input_window + steps), pred_output,color="blue", label='predictions')
        pyplot.grid(True, which='both')
        pyplot.title(type)
        pyplot.legend()

        if wandb_log:
            wandb.log({f"pred_{type}": data_tensor, f"true_{type}": data_true_future[-steps:]})

        sub_num += 1

    # num_types = len(data_source_list)
    # mse_result, mae_result, smape_result = total_mse / num_types, total_mae / num_types, smape_score / num_types

    mse_result = mse(total_pred, total_true)
    mae_result = mae((total_pred, total_true))
    smape_result = smape(total_pred, total_true)
    
    with open(RESULT_TXT_PATH, 'a') as f:
        f.write(f"Total {epoch} epochs - mse: {mse_result}, mae: {mae_result}, sampe: {smape_result}\n")
    if wandb_log:
        wandb.log({"pred_MSE": mse_result, "pred_MAE": mae_result, 'pred_SMAPE' : smape_result})

    pyplot.savefig(RESULT_PATH + '/future%d/transformer-future%d.png'%(origin_step, epoch))
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
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:].squeeze()).cpu().item()
        total_val_len += len(data_source)
    return total_loss / total_val_len