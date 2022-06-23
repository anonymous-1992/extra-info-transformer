import gzip

from optuna.samplers import TPESampler
from scipy.ndimage import gaussian_filter

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
from optuna.trial import TrialState


parser = argparse.ArgumentParser(description="train context-aware attention")
parser.add_argument("--name", type=str, default='basic_attn')
parser.add_argument("--exp_name", type=str, default='electricity')
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--total_steps", type=int, default=192)
parser.add_argument("--cuda", type=str, default='cuda:0')
parser.add_argument("--attn_type", type=str, default='basic_attn')
args = parser.parse_args()

n_distinct_trial = 1
config = ExperimentConfig(args.exp_name)
formatter = config.make_data_formatter()

data_csv_path = "{}.csv".format(args.exp_name)

print("Loading & splitting data_set...")
raw_data = pd.read_csv(data_csv_path)
train_data, valid, test = formatter.split_data(raw_data)
train_max, valid_max = formatter.get_num_samples_for_calibration()
params = formatter.get_experiment_params()

batch_size = 256

train_sample = batch_sampled_data(train_data, train_max, batch_size, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"], args.seed)

valid_sample = batch_sampled_data(valid, valid_max, batch_size, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"], args.seed)

test_sample = batch_sampled_data(test, valid_max, batch_size, args.total_steps,
                                     params['num_encoder_steps'], params["column_definition"], args.seed)


log_b_size = math.ceil(math.log2(batch_size))
log_s_size = math.ceil(math.log2(params['num_encoder_steps']))
param_history = list()

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Running on GPU")

model_params = formatter.get_default_model_params()
param_history = list()

criterion = nn.MSELoss()
mae = nn.L1Loss()

np.random.seed(args.seed)
random.seed(args.seed)
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


def define_model(d_model, n_heads,
                 stack_size, src_input_size,
                 tgt_input_size, kernel):

    d_k = int(d_model / n_heads)

    mdl = Attn(src_input_size=src_input_size,
               tgt_input_size=tgt_input_size,
               d_model=d_model,
               d_ff=d_model * 4,
               d_k=d_k, d_v=d_k, n_heads=n_heads,
               n_layers=stack_size, src_pad_index=0,
               tgt_pad_index=0, device=device,
               attn_type=args.attn_type, seed=args.seed,
               kernel=kernel)
    mdl.to(device)
    return mdl


def objective(trial):

    global best_model
    global val_loss
    global n_distinct_trial

    d_model = trial.suggest_categorical("d_model", [16, 32])
    if args.attn_type == "conv_attn":
        kernel = trial.suggest_categorical("kernel", [1, 3, 6, 9])
    else:
        kernel = trial.suggest_categorical("kernel", [1])

    if [d_model, kernel] in param_history or n_distinct_trial > 4:
        raise optuna.exceptions.TrialPruned()
    else:
        n_distinct_trial += 1
    param_history.append([d_model, kernel])
    stack_size = model_params["stack_size"]
    n_heads = model_params["num_heads"]

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

    model = define_model(d_model, n_heads, stack_size, train_en.shape[3], train_de.shape[3], kernel)

    optim = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 1000)

    best_iter_num = 0
    val_inner_loss = 1e10
    total_inner_loss = 1e10
    s = time.time()
    for epoch in range(params['num_epochs']):
        total_loss = 0
        model.train()
        for batch_id in range(train_en.shape[0]):
            output = model(train_en[batch_id], train_de[batch_id])
            '''smooth_output = torch.from_numpy(gaussian_filter(output.detach().cpu().numpy(), sigma=5)).to(device)
            loss = criterion(output, train_y[batch_id]) + 0.1 * L1Loss(output, smooth_output)'''
            loss = criterion(output, train_y[batch_id])
            total_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()

        print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(valid_en.shape[0]):
            outputs = model(valid_en[j], valid_de[j])
            loss = criterion(valid_y[j], outputs)
            test_loss += loss.item()

        if val_inner_loss > test_loss and total_inner_loss > total_loss:
            val_inner_loss = test_loss
            total_inner_loss = total_loss
            if val_loss > val_inner_loss:
                val_loss = val_inner_loss
                best_model = model
            best_iter_num = epoch

        print("Validation loss: {}".format(test_loss))

        end = time.time()

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch - best_iter_num >= 10:
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

    test_en, test_de, test_y, test_id = torch.from_numpy(test_sample['enc_inputs']), \
                                        torch.from_numpy(test_sample['dec_inputs']), \
                                        torch.from_numpy(test_sample['outputs']), \
                                        test_sample['identifier']

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

    len_s = test_y.shape[2]
    pred_path = "prediction_{}_{}".format(args.exp_name, len_s)
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    with gzip.open(os.path.join(pred_path, "{}_{}.json".format(args.name, str(args.seed))), 'w') as fout:
        fout.write(json.dumps(predictions.cpu().numpy().tolist()).encode('utf-8'))

    normaliser = targets_all.to(device).abs().mean()
    test_loss = criterion(predictions.to(device), targets_all.to(device)).item()
    test_loss = math.sqrt(test_loss) / normaliser.item()

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    mae_loss = mae_loss / normaliser.item()

    return test_loss, mae_loss


def main():

    '''search_space = {"d_model": [16, 32], "n_ext_info": [log_b_size*4, log_b_size],
                    "kernel_s":  [1, 3, 9, 15], "kernel_b": [1, 3, 9]}'''
    study = optuna.create_study(study_name=args.name,
                                direction="minimize", pruner=optuna.pruners.HyperbandPruner(),
                                sampler=TPESampler(seed=1234))
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
    key = "{}_{}".format(args.name, args.seed)
    error_file[key] = list()
    error_file[key].append("{:.3f}".format(nrmse))
    error_file[key].append("{:.3f}".format(nmae))

    res_path = "results_{}_{}.json".format(args.exp_name,
                                           args.total_steps - params['num_encoder_steps'])

    if os.path.exists(res_path):
        with open(res_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(key) is None:
                json_dat[key] = list()
            json_dat[key].append("{:.3f}".format(nrmse))
            json_dat[key].append("{:.3f}".format(nmae))

        with open(res_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(res_path, "w") as json_file:
            json.dump(error_file, json_file)


if __name__ == '__main__':
    main()