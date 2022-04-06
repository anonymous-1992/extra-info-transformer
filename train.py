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
parser.add_argument("--n_trials", type=int, default=6)
parser.add_argument("--total_steps", type=int, default=216)
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

train_sample = batch_sampled_data(train_data, train_max, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"])

valid_sample = batch_sampled_data(valid, valid_max, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"])

test_sample = batch_sampled_data(test, valid_max, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"])


batch_size = 512
log_b_size = math.ceil(math.log2(batch_size))
l_b_size = math.ceil(math.log2(batch_size))
param_history = list()

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Running on GPU")

model_params = formatter.get_default_model_params()
param_history = list()

criterion = nn.MSELoss()
mae = nn.L1Loss()

np.random.seed(21)
random.seed(21)

torch.manual_seed(args.seed)

val_loss = 1e10
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


def define_model(d_model, n_ext_info, n_heads, stack_size, src_input_size, tgt_input_size):

    d_k = int(d_model / n_heads)

    mdl = Attn(src_input_size=src_input_size,
                 tgt_input_size=tgt_input_size,
                 d_model=d_model,
                 d_ff=d_model * 4,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=stack_size, src_pad_index=0,
                 tgt_pad_index=0, device=device,
                attn_type=args.attn_type,
                n_ext_info=n_ext_info)
    mdl.to(device)
    return mdl


def objective(trial):

    global best_model
    global val_loss

    d_model = trial.suggest_categorical("d_model", [64, 16, 32])
    if args.attn_type == "extra_info_attn":
        n_ext_info = trial.suggest_categorical("n_ext_info", [log_b_size, log_b_size*4, log_b_size*8])
    else:
        n_ext_info = 0
    if [d_model, n_ext_info] in param_history:
        raise optuna.exceptions.TrialPruned()
    param_history.append([d_model, n_ext_info])
    n_heads = model_params["num_heads"]
    stack_size = model_params["stack_size"]

    train_en, train_de, train_y, train_id = torch.from_numpy(train_sample['enc_inputs']).to(device), \
                                            torch.from_numpy(train_sample['dec_inputs']).to(device), \
                                            torch.from_numpy(train_sample['outputs']).to(device), \
                                            train_sample['identifier']

    valid_en, valid_de, valid_y, valid_id = torch.from_numpy(valid_sample['enc_inputs']).to(device), \
                                            torch.from_numpy(valid_sample['dec_inputs']).to(device), \
                                            torch.from_numpy(valid_sample['outputs']).to(device), \
                                            valid_sample['identifier']

    train_en, train_de, train_y, train_id = batching(batch_size, train_en,
                                                             train_de, train_y, train_id)
    train_en, train_de, train_y, train_id = \
        train_en.to(device), train_de.to(device), train_y.to(device), train_id

    valid_en, valid_de, valid_y, valid_id = batching(batch_size, valid_en,
                                                             valid_de, valid_y, valid_id)

    valid_en, valid_de, valid_y, valid_id = \
        valid_en.to(device), valid_de.to(device), valid_y.to(device), valid_id

    model = define_model(d_model, n_ext_info, n_heads, stack_size, train_en.shape[3], train_de.shape[3])

    optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 4000)

    best_iter_num = 0
    val_inner_loss = 1e10
    for epoch in range(params['num_epochs']):
        total_loss = 0
        model.train()
        for batch_id in range(train_en.shape[0]):
            output = model(train_en[batch_id], train_de[batch_id])
            '''smooth_output = torch.from_numpy(gaussian_filter(output.detach().cpu().numpy(), sigma=5)).to(device)
            loss = criterion(output, train_y_p[batch_id]) + lam * L1Loss(output, smooth_output)'''
            loss = criterion(output, train_y[batch_id])
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()

        print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(valid_en.shape[0]):
            outputs = model(valid_en[j], valid_de[j])
            loss = criterion(valid_y[j], outputs)
            test_loss += loss.item()

        if val_inner_loss > test_loss:
            val_inner_loss = test_loss
            if val_loss > val_inner_loss:
                val_loss = val_inner_loss
                best_model = model
            best_iter_num = epoch

        print("Validation loss: {}".format(test_loss))

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch - best_iter_num >= 15:
            break

    return val_inner_loss


def evaluate():

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    model = best_model
    model.eval()

    sample_data = batch_sampled_data(test, valid_max, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"])
    test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']), \
                                        torch.from_numpy(sample_data['dec_inputs']), \
                                        torch.from_numpy(sample_data['outputs']), \
                                        sample_data['identifier']

    test_en, test_de, test_y, test_id = batching(batch_size, test_en,
                                                 test_de, test_y, test_id)

    test_en, test_de, test_y, test_id = \
        test_en.to(device), test_de.to(device), test_y.to(device), test_id

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


def main():

    search_space = {"d_model": [16, 32, 64], "n_ext_info": [log_b_size*8, log_b_size*4, log_b_size]}
    study = optuna.create_study(study_name=args.name,
                                direction="minimize", pruner=optuna.pruners.HyperbandPruner(),
                                sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=args.n_trials)

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
                                           args.total_steps - params['num_encoder_steps'])

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