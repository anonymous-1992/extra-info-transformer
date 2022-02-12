import argparse
import json
from models.context_aware_attn import Attn
import torch.nn as nn
import torch
import os
from base_train import batching, batch_sampled_data, inverse_output
import math
import pandas as pd
from data.data_loader import ExperimentConfig


def evaluate(config, args, test_en, test_de, test_y, test_id, formatter, path, device):

    batch_size, n_heads, d_model = config
    d_k = int(d_model / n_heads)
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    model_params = formatter.get_default_model_params()

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    model = Attn(src_input_size=test_en.shape[3],
                 tgt_input_size=test_de.shape[3],
                 d_model=d_model,
                 d_ff=d_model * 4,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=model_params['stack_size'], src_pad_index=0,
                 tgt_pad_index=0, device=device,  context_lengths=model_params['context_lengths']).to(device)

    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    predictions = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])
    targets_all = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])

    for j in range(test_en.shape[0]):
        output = model(test_en[j], test_de[j])
        output_map = inverse_output(output, test_y[j], test_id[j])
        forecast = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32')).to(device)

        predictions[j, :, :] = forecast
        targets = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')).to(device)

        targets_all[j, :, :] = targets

    test_loss = mse(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    test_loss = math.sqrt(test_loss) / normaliser

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    mae_loss = mae_loss / normaliser

    return test_loss, mae_loss


def main():

    parser = argparse.ArgumentParser(description="train context-aware attention")
    parser.add_argument("--name", type=str, default='context-aware-attn')
    parser.add_argument("--exp_name", type=str, default='watershed')
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()
    error_file = dict()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Running on GPU")

    data_csv_path = "{}.csv".format(args.exp_name)

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    _, _, test = formatter.split_data(raw_data)
    _, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                        torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                        torch.from_numpy(sample_data['outputs']).to(device), \
                                        sample_data['identifier']

    path = "models_{}_{}".format(args.exp_name, params['total_time_steps'] - params['num_encoder_steps'])

    test_en_b, test_de_b, test_y_b, test_id_b = batching(params['minibatch_size'], test_en,
                                                         test_de, test_y, test_id)

    with open('configs_{}_{}.json'.format(args.exp_name,
             params['total_time_steps'] - params['num_encoder_steps']), 'r') as json_file:
        configs = json.load(json_file)

    nrmse, nmae = evaluate(configs, args, test_en_b, test_de_b, test_y_b, test_id_b, formatter, path, device)

    error_file[args.name] = list()
    error_file[args.name].append(nrmse)
    error_file[args.name].append(nmae)

    config_path = "configs_{}_{}.json".format(args.exp_name,
                                              params['total_time_steps'] - params['num_encoder_steps'])

    if os.path.exists(config_path):
        with open(config_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                json_dat[args.name] = list()
            json_dat[args.name].append(nrmse)
            json_dat[args.name].append(nmae)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(error_file, json_file)