# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
from itertools import chain
from itertools import groupby

import torch
import numpy as np
from Utils import utils, base
import pandas as pd
import math
import random

InputTypes = base.InputTypes


def batching(batch_size, x_en, x_de, y_t, test_id):

    batch_n = math.floor(x_en.shape[0] / batch_size)
    start = 0
    X_en = torch.zeros(batch_n, batch_size, x_en.shape[1], x_en.shape[2])
    X_de = torch.zeros(batch_n, batch_size, x_de.shape[1], x_de.shape[2])
    Y_t = torch.zeros(batch_n, batch_size, y_t.shape[1], y_t.shape[2])
    tst_id = np.empty((batch_n, batch_size, test_id.shape[1], x_en.shape[2]), dtype=object)
    i = 0
    while start+batch_size <= x_en.shape[0]:

        X_en[i, :, :, :] = x_en[start:start+batch_size, :, :]
        X_de[i, :, :, :] = x_de[start:start+batch_size, :, :]
        Y_t[i, :, :, :] = y_t[start:start+batch_size, :, :]
        tst_id[i, :, :, :] = test_id[start:start+batch_size, :, :]
        start += batch_size
        i += 1

    return X_en, X_de, Y_t, tst_id


def batch_sampled_data(data, max_samples, batch_size, time_steps,
                       num_encoder_steps, column_definition, seed):
    """Samples segments into a compatible format.
    Args:
      data: Sources data_set to sample and batch
      max_samples: Maximum number of samples in batch
    Returns:
      Dictionary of batched data_set with the maximum samples specified.
    """

    np.random.seed(seed)
    random.seed(seed)

    if max_samples < 1:
        raise ValueError(
          'Illegal number of samples specified! samples={}'.format(max_samples))

    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)
    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)

    data.sort_values(by=[id_col, time_col], inplace=True)

    valid_sampling_locations = []
    split_data_map = {}
    for identifier, df in data.groupby(id_col):
        num_entries = len(df)
        if num_entries >= time_steps:
            valid_sampling_locations += [
                (identifier, time_steps + i)
                for i in range(num_entries - time_steps + 1)
            ]

            split_data_map[identifier] = df

    def takeSecond(elem):
        return elem[0], elem[1]

    if 0 < max_samples < len(valid_sampling_locations):

        ranges = [
          valid_sampling_locations[i] for i in np.random.choice(
              len(valid_sampling_locations), max_samples, replace=False)
        ]

    else:
        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), len(valid_sampling_locations), replace=False)
        ]

    '''ranges.sort(key=lambda elem: (elem[0], elem[1]))
    ranges = [[x for x in g] for k, g in groupby(ranges, key=lambda x:x[0])]
    chunk_size = int(math.log2(batch_size))
    ranges = list(chain.from_iterable(ranges))
    ranges = [ranges[i:i+chunk_size] for i in range(0, len(ranges), chunk_size)]
    random.shuffle(ranges)
    ranges = list(chain.from_iterable(ranges))'''

    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)
    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)
    target_col = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definition)
    enc_input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]
    dec_input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
    ]
    input_size = len(enc_input_cols)
    inputs = np.zeros((max_samples, time_steps, input_size))
    enc_inputs = np.zeros((max_samples, num_encoder_steps, input_size))
    dec_inputs = np.zeros((max_samples, time_steps - num_encoder_steps, input_size - 1))
    outputs = np.zeros((max_samples, time_steps, 1))
    time = np.empty((max_samples, time_steps, 1), dtype=object)
    identifiers = np.empty((max_samples, time_steps, 1), dtype=object)

    for i, tup in enumerate(ranges):
        if (i + 1 % 1000) == 0:
            print(i + 1, 'of', max_samples, 'samples done...')
        identifier, start_idx = tup
        sliced = split_data_map[identifier].iloc[start_idx -
                                               time_steps:start_idx]
        enc_inputs[i, :, :] = sliced[enc_input_cols].iloc[:num_encoder_steps]
        dec_inputs[i, :, :] = sliced[dec_input_cols].iloc[num_encoder_steps:]
        inputs[i, :, :] = sliced[enc_input_cols]
        outputs[i, :, :] = sliced[[target_col]]
        time[i, :, 0] = sliced[time_col]
        identifiers[i, :, 0] = sliced[id_col]

    sampled_data = {
        'inputs': inputs,
        'enc_inputs': enc_inputs,
        'dec_inputs': dec_inputs,
        'outputs': outputs[:, num_encoder_steps:, :],
        'active_entries': np.ones_like(outputs[:, num_encoder_steps:, :]),
        'time': time,
        'identifier': identifiers
    }

    return sampled_data


def inverse_output(predictions, outputs, test_id):

    def format_outputs(preds):
        flat_prediction = pd.DataFrame(
            preds[:, :, 0],
            columns=[
                't+{}'.format(i)
                for i in range(preds.shape[1])
            ]
        )
        flat_prediction['identifier'] = test_id[:, 0, 0]
        return flat_prediction

    process_map = {'predictions': format_outputs(predictions), 'targets': format_outputs(outputs)}
    return process_map
