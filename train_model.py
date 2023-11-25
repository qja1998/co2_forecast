import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from models import Model
from dataset import make_dataset


def train(model, train_loader, val_data, optimizer, criterion, epoch):
    model.train()
    for i in (range(epoch)):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.squeeze(-1).to(device)
            y = y.unsqueeze(-1).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, epoch,
                100. * i / epoch, loss.item()))

    if val_data is not None:
        model.eval()
        loss = 0
        true_val = []
        predictions = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_data):
                
                x = x.view(1, -1).to(device)
                y = y.unsqueeze(-1).to(device)
                output = model(x).squeeze(-1)

                loss += criterion(output, y)
                true_val.append(float(y.cpu().numpy()))
                predictions.append(float(output.cpu().numpy()))
        

        print('\nTest set: Average loss: {:.4f}'.format(loss / len(val_data)))
        # print(true_val, predictions)
        plt.figure(figsize=(30,10))
        x = np.arange(len(true_val))
        plt.plot(x, true_val, label='true', c='blue')
        plt.plot(x, predictions, label='predictions', c='red')
        plt.legend()
        plt.savefig(f'./results/{model_name}-{epoch}-{loss / len(val_data):.4f}.png')
    
        return loss / len(val_data)
    
    return loss / len(train_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="lstm, gru, rnn train")

    parser.add_argument('--model_name', dest='model_name', help='rnn|lstm|gru', type=str)
    parser.add_argument('--seq_len', dest='seq_len',type=int, default=7)
    parser.add_argument('--hidden_szie', dest='hidden_szie', type=int, default=128)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=1)
    parser.add_argument('--conv_size', dest='conv_size', type=int, default=None)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_szie', dest='batch_szie', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=300)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1)
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--train_rate', dest='train_rate', type=float, default=0.8)
    parser.add_argument('--sup_type', dest='sup_type', type=str, default='A')
    parser.add_argument('--year', dest='year', type=int, default=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    seq_len = args.seq_len
    hidden_szie = args.hidden_szie
    num_layers = args.num_layers
    conv_size = args.conv_size
    learning_rate = args.learning_rate
    batch_szie = args.batch_szie
    epochs = args.epochs
    data_path = args.data_path
    train_rate = args.train_rate
    sup_type = args.sup_type
    year = args.year

    df = pd.read_csv(data_path)

    train_dataset, val_dataset = make_dataset(df, seq_len, train_rate, sup_type, year)
    model = Model(model_name, seq_len, hidden_szie, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss().to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_szie, shuffle=True)

    loss = train(model, train_loader, val_dataset, optimizer, criterion, epochs)
    torch.save(model.state_dict(), f"saved_models/{model_name}-{epochs}-{loss}.pth")
