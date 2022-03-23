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
from Utils.base_train import batching, batch_sampled_data, inverse_output
import time
import optuna
import math
from scipy.ndimage import gaussian_filter
from optuna.trial import TrialState


parser = argparse.ArgumentParser(description="train context-aware attention")
parser.add_argument("--name", type=str, default='extra_info_attn')
parser.add_argument("--exp_name", type=str, default='electricity')
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cuda", type=str, default='cuda:0')
parser.add_argument("--attn_type", type=str, default='extra_info_attn')
args = parser.parse_args()

config = ExperimentConfig(args.exp_name)
formatter = config.make_data_formatter()

data_csv_path = "{}.csv".format(args.exp_name)

print("Loading & splitting data...")
raw_data = pd.read_csv(data_csv_path)
train_data, valid, test = formatter.split_data(raw_data)
train_max, valid_max = formatter.get_num_samples_for_calibration()
params = formatter.get_experiment_params()

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Running on GPU")

model_params = formatter.get_default_model_params()

criterion = nn.MSELoss()
mae = nn.L1Loss()

batch_size = model_params["minibatch_size"][0]

np.random.seed(1234)
random.seed(1234)

torch.manual_seed(args.seed)

mdl = nn.Module()
best_model = nn.Module()


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
L1Loss = nn.L1Loss()


def define_model(d_model, n_heads, stack_size, src_input_size, tgt_input_size):

    global mdl
    d_k = int(d_model / n_heads)

    model = Attn(src_input_size=src_input_size,
                 tgt_input_size=tgt_input_size,
                 d_model=d_model,
                 d_ff=d_model * 4,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=stack_size, src_pad_index=0,
                 tgt_pad_index=0, device=device, attn_type=args.attn_type)
    model.to(device)
    mdl = model
    return model


def callback(study, trial):
    global best_model
    if study.best_trial == trial:
        best_model = mdl


def objective(trial):

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

    train_en_p, train_de_p, train_y_p, train_id_p = batching(batch_size, train_en,
                                                             train_de, train_y, train_id)
    train_en_p, train_de_p, train_y_p, train_id_p = \
        train_en_p.to(device), train_de_p.to(device), train_y_p.to(device), train_id_p

    valid_en_p, valid_de_p, valid_y_p, valid_id_p = batching(batch_size, valid_en,
                                                             valid_de, valid_y, valid_id)

    valid_en_p, valid_de_p, valid_y_p, valid_id_p = \
        valid_en_p.to(device), valid_de_p.to(device), valid_y_p.to(device), valid_id_p

    seq_len = params['total_time_steps'] - params['num_encoder_steps']
    path = "models_{}_{}".format(args.exp_name, seq_len)
    if not os.path.exists(path):
        os.makedirs(path)

    d_model = trial.suggest_categorical("d_model", [16, 32])
    loss_func = trial.suggest_categorical("loss_func", ["mseLoss", "smoothMseLoss"])
    n_heads = model_params["num_heads"]
    stack_size = model_params["stack_size"]
    lam = 0.1

    model = define_model(d_model, n_heads, stack_size, train_en_p.shape[3], train_de_p.shape[3])

    optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 4000)

    total_loss = 0
    for epoch in range(params['num_epochs']):
        model.train()
        for batch_id in range(train_en_p.shape[0]):
            output = model(train_en_p[batch_id], train_de_p[batch_id])
            if loss_func == "mseLoss":
                loss = criterion(output, train_y_p[batch_id])
            else:
                smooth_output = torch.from_numpy(gaussian_filter(output.detach().cpu().numpy(), sigma=1)).to(device)
                loss = criterion(output, train_y_p[batch_id]) + lam * L1Loss(output, smooth_output)

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()

        print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(valid_en_p.shape[0]):
            outputs = model(valid_en_p[j], valid_de_p[j])
            loss = criterion(valid_y_p[j], outputs)
            test_loss += loss.item()

        trial.report(test_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_loss


def evaluate():

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']), \
                                        torch.from_numpy(sample_data['dec_inputs']), \
                                        torch.from_numpy(sample_data['outputs']), \
                                        sample_data['identifier']

    test_en, test_de, test_y, test_id = batching(batch_size, test_en,
                                                         test_de, test_y, test_id)

    test_en, test_de, test_y, test_id = \
        test_en.to(device), test_de.to(device), test_y.to(device), test_id
    model = best_model
    model.eval()

    predictions = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])
    targets_all = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])
    for j in range(test_en.shape[0]):
        output = model(test_en[j], test_de[j])
        output_map = inverse_output(output, test_y[j], test_id[j])
        forecast = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32')).to(device)

        predictions[j, :forecast.shape[0], :] = forecast
        targets = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')).to(device)

        targets_all[j, :targets.shape[0], :] = targets

    normaliser = targets_all.to(device).abs().mean()
    test_loss = criterion(predictions.to(device), targets_all.to(device)).item()
    test_loss = math.sqrt(test_loss) / normaliser.item()

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    mae_loss = mae_loss / normaliser.item()

    return test_loss, mae_loss


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e, val_loss,
          val_inner_loss, optimizer, train_loss_list,
          config, config_num, best_config, criterion, path, stop, device):


    lam = 0.1
    try:
        model.train()
        total_loss = 0
        start = time.time()
        for batch_id in range(train_en.shape[0]):
            output = \
                model(train_en[batch_id], train_de[batch_id])
            '''smooth_output = torch.from_numpy(gaussian_filter(output.detach().cpu().numpy(), sigma=1))\
                .to(device)'''
            #loss = criterion(output, train_y[batch_id]) + lam * L1Loss(output, smooth_output)
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
                torch.save({'model_state_dict': model.state_dict()},
                           os.path.join(path, "{}_{}".format(args.name, args.seed)))

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

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4, timeout=600, callbacks=[callback])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    nrmse, nmae = evaluate()

    error_file = dict()
    error_file[args.name] = list()
    error_file[args.name].append("{:.3f}".format(nrmse))
    error_file[args.name].append("{:.3f}".format(nmae))

    res_path = "results_{}_{}.json".format(args.exp_name,
                                           params['total_time_steps'] - params['num_encoder_steps'])

    if os.path.exists(res_path):
        with open(res_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                json_dat[args.name] = list()
            json_dat[args.name].append("{:.3f}".format(nrmse))
            json_dat[args.name].append("{:.3f}".format(nmae))

        with open(res_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(res_path, "w") as json_file:
            json.dump(error_file, json_file)


if __name__ == '__main__':
    main()