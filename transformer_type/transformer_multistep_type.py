import os
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

from dataset import get_data
from model import TransAm, train, predict_future, evaluate

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


def train_start(input_window, output_window, batch_size, epochs, feature_size, d_ff, num_layers, lr, dropout=0.1, is_save=False):
    # Set parameters
    input_window = input_window
    output_window = output_window
    batch_size = batch_size # batch size
    best_val_loss = float("inf")
    epochs = epochs # The number of epochs
    log_epoch = 15
    best_models = dict()
    pred_step = 180

    feature_size = feature_size
    num_layers = num_layers
    d_ff = d_ff
    lr = lr
    dropout = dropout

    RESULT_PATH = f"./transformer_results/type_{input_window}-{output_window}_{batch_size}_{feature_size}-{d_ff}-{num_layers}_{lr}_{epochs}_{dropout}"
    RESULT_TXT_PATH = RESULT_PATH + "/output.txt"

    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    if not os.path.isdir(RESULT_PATH + f"/future{pred_step}"):
        os.mkdir(RESULT_PATH + f"/future{pred_step}")
    if not os.path.isdir(RESULT_PATH + f"/models"):
        os.mkdir(RESULT_PATH + f"/models")
    with open(RESULT_TXT_PATH, 'w') as f:
        f.write('')

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

    type_list = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I']
    train_data_list = []
    val_data_list = []

    for type_name in type_list:
        train_data, val_data, scaler = get_data(type_name, input_window, output_window)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        train_data_list.append((train_data, type_name, scaler))
        val_data_list.append((val_data, type_name, scaler))

    criterion = nn.MSELoss()


    models = {type:[TransAm(feature_size=feature_size, num_layers=num_layers, d_ff=d_ff, dropout=dropout).to(device)] for _, type, _ in train_data_list}

    for t in models:
        model = models[t][0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
        models[t].extend([optimizer, scheduler])


    train_losses = []
    val_losses = []
    num_types = len(train_data_list)

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_start_time = time.time()
        total_train_loss = 0
        total_val_loss = 0

        for train_data, type, scaler in train_data_list:
            model, optimizer, scheduler = models[type]
            model, optimizer, scheduler, loss = train(model,
                                                      train_data,
                                                      type,
                                                      epoch,
                                                      optimizer,
                                                      scheduler,
                                                      criterion,
                                                      batch_size,
                                                      input_window,
                                                      output_window,
                                                      RESULT_TXT_PATH)
            models[type] = model, optimizer, scheduler
            total_train_loss += loss
        
        train_losses.append(total_train_loss / num_types)

        if(epoch % log_epoch == 0):

            pred_mse, pred_mae, pred_smape = predict_future(models,
                                             val_data_list, epoch,
                                             pred_step,
                                             input_window,
                                             output_window,
                                             RESULT_TXT_PATH,
                                             RESULT_PATH)
            # save loss graph
            pyplot.plot(train_losses, label='train loss')
            pyplot.plot(val_losses, label='val loss')
            pyplot.yscale('log')
            pyplot.legend()
            pyplot.savefig(RESULT_PATH + f"/loss.png")
            pyplot.close()

            if val_loss < best_val_loss:
                best_val_loss = 2 * pred_smape + pred_mse
                best_models[t] = model


        for val_data, type, scaler in val_data_list:
            model, optimizer, scheduler = models[type]
            val_loss = evaluate(model, val_data, criterion, input_window, output_window)
            wandb.log({f"val_loss_{type}": val_loss})
            total_val_loss += val_loss
        val_losses.append(total_val_loss / num_types)
        total_train_loss = total_val_loss / num_types
        with open(RESULT_TXT_PATH, 'a') as f:
            f.write('-' * 89 + '\n')
            f.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss(mse) {:5.5f} | valid ppl {:8.2f}\n'.format(epoch, (time.time() - epoch_start_time),
                                            total_train_loss, math.exp(total_train_loss)))
            f.write('-' * 89 + '\n')
            wandb.log({"val_loss_total": total_train_loss})

        scheduler.step()

    if is_save:
        for t in best_models:
            model, _, _ = best_models[t]
            torch.save(model.state_dict(), f"{RESULT_PATH}/models/{t}_{input_window}_{batch_size}_{feature_size}-{num_layers}_{lr}_{epoch}-{best_val_loss:.5f}_{val_loss:.5f}.pt")