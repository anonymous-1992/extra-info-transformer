from models.extra_info_atttn import Attn
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import itertools
import sys
import random
import pandas as pd
from data.data_loader import ExperimentConfig
from Utils.base_train import batching, batch_sampled_data
import time

class NoamOpt:

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


torch.autograd.set_detect_anomaly(True)


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e, val_loss,
          val_inner_loss, optimizer, train_loss_list,
          config, config_num, best_config, criterion, path, stop):

    try:
        model.train()
        total_loss = 0
        start = time.time()
        for batch_id in range(train_en.shape[0]):
            output = \
                model(train_en[batch_id], train_de[batch_id])
            loss = criterion(output, train_y[batch_id])
            train_loss_list.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
        end = time.time()
        total_time = end - start
        print("total time: {}".format(total_time))

        print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(test_en.shape[0]):
            outputs = model(test_en[j], test_de[j])
            loss = criterion(test_y[j], outputs)
            test_loss += loss.item()

        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                val_loss = val_inner_loss
                best_config = config
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

            e = epoch

        if epoch - e > 5:
            stop = True

        print("Average loss: {:.4f}".format(test_loss))

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config_num': config_num,
            'best_config': best_config
        }, os.path.join(path, "{}_continue".format(args.name)))
        sys.exit(0)

    return best_config, val_loss, val_inner_loss, stop, e


def create_config(hyper_parameters):
    prod = list(itertools.product(*hyper_parameters))
    return list(random.sample(set(prod), len(prod)))


def main():

    parser = argparse.ArgumentParser(description="train context-aware attention")
    parser.add_argument("--name", type=str, default='extra-info-attn')
    parser.add_argument("--exp_name", type=str, default='electricity')
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--cuda", type=str, default='cuda:0')
    args = parser.parse_args()
    config_file = dict()

    np.random.seed(21)
    random.seed(21)

    torch.manual_seed(args.seed)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Running on GPU")

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    data_csv_path = "{}.csv".format(args.exp_name)

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    sample_data = batch_sampled_data(train_data, train_max, params['total_time_steps'],
                       params['num_encoder_steps'], params["column_definition"])
    train_en, train_de, train_y, train_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    sample_data = batch_sampled_data(valid, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    valid_en, valid_de, valid_y, valid_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    model_params = formatter.get_default_model_params()

    seq_len = params['total_time_steps'] - params['num_encoder_steps']
    path = "models_{}_{}".format(args.exp_name, seq_len)
    if not os.path.exists(path):
        os.makedirs(path)

    criterion = nn.MSELoss()
    hyper_param = list([model_params['minibatch_size'], [model_params['num_heads']],
                        model_params['hidden_layer_size']])
    configs = create_config(hyper_param)
    print('number of config: {}'.format(len(configs)))

    val_loss = 1e10
    best_config = configs[0]
    config_num = 0

    for i, conf in enumerate(configs, config_num):
        print('config {}: {}'.format(i+1, conf))

        batch_size, n_heads, d_model = conf
        d_k = int(d_model / n_heads)

        train_en_p, train_de_p, train_y_p, train_id_p = batching(batch_size, train_en,
                                                         train_de, train_y, train_id)

        valid_en_p, valid_de_p, valid_y_p, valid_id_p = batching(batch_size, valid_en,
                                                         valid_de, valid_y, valid_id)

        model = Attn(src_input_size=train_en_p.shape[3],
                     tgt_input_size=train_de_p.shape[3],
                     d_model=d_model,
                     d_ff=d_model*4,
                     d_k=d_k, d_v=d_k, n_heads=n_heads,
                     n_layers=model_params['stack_size'], src_pad_index=0,
                     tgt_pad_index=0, device=device)
        model.to(device)

        optim = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 4000)

        epoch_start = 0

        val_inner_loss = 1e10
        train_loss_list = list()
        stop = False

        for epoch in range(epoch_start, params['num_epochs'], 1):

            best_config, val_loss, val_inner_loss, stop, e = \
                train(args, model, train_en_p.to(device), train_de_p.to(device),
                      train_y_p.to(device), valid_en_p.to(device), valid_de_p.to(device),
                      valid_y_p.to(device), epoch, params['num_epochs'], val_loss, val_inner_loss,
                      optim, train_loss_list, conf, i, best_config, criterion, path, stop)

            if stop:
                break

    batch_size, heads, d_model = best_config
    print("best_config: {}".format(best_config))

    config_file[args.name] = list()
    config_file[args.name].append(batch_size)
    config_file[args.name].append(heads)
    config_file[args.name].append(d_model)

    config_path = "configs_{}_{}.json".format(args.exp_name, seq_len)

    if os.path.exists(config_path):
        with open(config_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                json_dat[args.name] = list()
            json_dat[args.name].append(batch_size)
            json_dat[args.name].append(heads)
            json_dat[args.name].append(d_model)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(config_file, json_file)


if __name__ == '__main__':
    main()