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

import wget
import pyunpack
import os
import pandas as pd
import numpy as np
import argparse
import sys
import random
import gc
import glob
import datetime

from data import electricity, traffic, watershed

np.random.seed(21)
random.seed(21)


class ExperimentConfig(object):
    default_experiments = ['electricity', 'traffic', 'air_quality', 'favorita', 'watershed', 'solar', 'camel']

    def __init__(self, experiment='electricity', root_folder=None):

        if experiment not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(experiment))

        if root_folder is None:
            root_folder = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '../', 'outputs')
            print('Using root folder {}'.format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, '', experiment)

        for relevant_directory in [
            self.root_folder, self.data_folder
        ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            'electricity': 'hourly_electricity.csv',
            'traffic': 'hourly_traffic.csv',
            'air_quality': 'hourly_air_quality.csv',
            'favorita': 'favorita_consolidated.csv',
            'watershed': 'watershed.csv',
            'solar': 'solar.csv',
            'camel': 'camel.csv'
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    def make_data_formatter(self):
        """Gets a data formatter object for experiment.
        Returns:
          Default DataFormatter per experiment.
        """

        data_formatter_class = {
            'electricity': electricity.ElectricityFormatter,
            'traffic': traffic.TrafficFormatter,
            'watershed': watershed.WatershedFormatter,
        }

        return data_formatter_class[self.experiment]()


def download_from_url(url, output_path):
    """Downloads a file from url."""

    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')


def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    '''if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))'''


def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')


def process_watershed(config):

    """Process watershed dataset
    Args:
    config: Default experiment config for Watershed
    """
    sites = ['BDC', 'BEF', 'DCF', 'GOF', 'HBF', 'LMP', 'MCQ', 'SBM', 'TPB', 'WHB']
    data_path = config.data_folder
    df_list = []

    for i, site in enumerate(sites):

        df = pd.read_csv('{}/{}_WQual_Level4.csv'.format(data_path, site), index_col=0, sep=',')
        df_list.append(df.iloc[:, :])

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output.Date)
    output.sort_index(inplace=True)
    output = output.dropna(axis=1, how='all')
    output = output.fillna(method="ffill").fillna(method='bfill')

    start_date = pd.to_datetime('2013-03-28')
    earliest_time = start_date
    output = output[output.index >= start_date]
    date = pd.to_datetime(output.index)
    output['day_of_week'] = date.dayofweek
    output['d'] = ["{}{}{}{}{}{}".format(str(x)[0:4], str(x)[5:7], str(x)[8:10],
                                         str(x)[11:13], str(x)[14:16], str(x)[17:19]) for x in date.values]
    output['hour'] = date.hour
    output['id'] = output['Site']
    output['categorical_id'] = output['Site']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24

    output['days_from_start'] = (date - earliest_time).days
    output = output[output['Site'] != 0.0]
    output = output.fillna('na')
    output = output[output['days_from_start'] != 'na']
    output.to_csv("watershed.csv")

    print('Done.')


def download_camel(args):

    """Downloads camel dataset"""
    url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"
    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, '')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)


def download_electricity(args):
    """Downloads electricity dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'LD2011_2014.txt')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data')

    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # used to determine the start and end dates of a series
    output = df.resample('1h').mean().replace(0., np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print('Processing {}'.format(label))
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
                date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                    & (output['days_from_start'] < 1346)].copy()

    output.to_csv("electricity.csv".format(args.data_folder))

    print('Done.')


def download_traffic(args):
    """Downloads traffic dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'PEMS_train')
    zip_path = os.path.join(data_folder, 'PEMS-SF.zip')

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data')

    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
          l = [
              variable_type(i) for i in s.replace('[', '').replace(']', '').split()
          ]
        else:
          l = [
              variable_type(i)
              for i in s.replace('[', '').replace(']', '').split(delimiter)
          ]

        return l

    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), 'r') as dat:
          l = process_list(dat.readlines()[0])
        return l

    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(data_folder, filename), 'r') as dat:

          lines = dat.readlines()
          for i, line in enumerate(lines):
            if (i + 1) % 50 == 0:
              print('Completed {} of {} rows for {}'.format(i + 1, len(lines),
                                                            filename))

            array = [
                process_list(row_split, variable_type=float, delimiter=None)
                for row_split in process_list(
                    line, variable_type=str, delimiter=';')
            ]
            array_list.append(array)

        return array_list

    shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
    train_dayofweek = read_single_list('PEMS_trainlabels')
    train_tensor = read_matrix('PEMS_train')
    test_dayofweek = read_single_list('PEMS_testlabels')
    test_tensor = read_matrix('PEMS_test')

    # Inverse permutate shuffle order
    print('Shuffling')
    inverse_mapping = {
      new_location: previous_location
      for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array([
      inverse_mapping[new_location]
      for new_location, _ in enumerate(shuffle_order)
    ])

    # Group and reoder based on permuation matrix
    print('Reodering')
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    # Put everything back into a dataframe
    print('Parsing as dataframe')
    labels = ['traj_{}'.format(i) for i in read_single_list('stations_list')]

    hourly_list = []
    for day, day_matrix in enumerate(combined_tensor):

        # Hourly data
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                                ]  # sampled at 10 min intervals
        if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
          raise ValueError('Invalid hour! {}-{}'.format(
              hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))

        hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
        hourly['sensor_day'] = day
        hourly['time_on_day'] = hourly.index
        hourly['day_of_week'] = day_of_week[day]

        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

    # Flatten such that each entitiy uses one row in dataframe
    store_columns = [c for c in hourly_frame.columns if 'traj' in c]
    other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
    flat_df = pd.DataFrame(columns=['values', 'prev_values', 'next_values'] +
                         other_columns + ['id'])

    def format_index_string(x):
        """Returns formatted string for key."""

        if x < 10:
            return '00' + str(x)
        elif x < 100:
            return '0' + str(x)
        elif x < 1000:
            return str(x)

        raise ValueError('Invalid value of x {}'.format(x))

    for store in store_columns:
        print('Processing {}'.format(store))

        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ['values'] + other_columns
        sliced['id'] = int(store.replace('traj_', ''))

        # Sort by Sensor-date-time
        key = sliced['id'].apply(str) \
          + sliced['sensor_day'].apply(lambda x: '_' + format_index_string(x)) \
            + sliced['time_on_day'].apply(lambda x: '_' + format_index_string(x))
        sliced = sliced.set_index(key).sort_index()

        sliced['values'] = sliced['values'].fillna(method='ffill')
        sliced['prev_values'] = sliced['values'].shift(1)
        sliced['next_values'] = sliced['values'].shift(-1)

        flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)

    # Filter to match range used by other academic papers
    index = flat_df['sensor_day']
    flat_df = flat_df[index < 173].copy()

    # Creating columns fo categorical inputs
    flat_df['categorical_id'] = flat_df['id'].copy()
    flat_df['hours_from_start'] = flat_df['time_on_day'] \
      + flat_df['sensor_day']*24.
    flat_df['categorical_day_of_week'] = flat_df['day_of_week'].copy()
    flat_df['categorical_time_on_day'] = flat_df['time_on_day'].copy()

    flat_df.to_csv("traffic.csv")
    print('Done.')


def main(expt_name, force_download, output_folder):

    print('#### Running download script ###')
    expt_config = ExperimentConfig(expt_name, output_folder)

    if os.path.exists(expt_config.data_csv_path) and not force_download:
        print('Data has been processed for {}. Skipping download...'.format(
            expt_name))
        sys.exit(0)
    else:
        print('Resetting data folder...')
        #shutil.rmtree(expt_config.data_csv_path)
        os.makedirs(expt_config.data_csv_path)

    # Default download functions
    download_functions = {
        'electricity': download_electricity,
        'traffic': download_traffic,
        'watershed': process_watershed,
        'camel': download_camel
    }

    if expt_name not in download_functions:
        raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

    download_function = download_functions[expt_name]

    # Run data download
    print('Getting {} data...'.format(expt_name))
    download_function(expt_config)

    print('Download completed.')


if __name__ == '__main__':
    def get_args():
        """Returns settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description='Data download configs')
        parser.add_argument(
            '--expt_name',
            type=str,
            nargs='?',
            choices=experiment_names,
            help='Experiment Name. Default={}'.format(','.join(experiment_names)))
        parser.add_argument(
            '--output_folder',
            type=str,
            nargs='?',
            default='.',
            help='Path to folder for data download')
        parser.add_argument(
            '--force_download',
            type=str,
            nargs='?',
            choices=['yes', 'no'],
            default='yes',
            help='Whether to re-run data download')

        args = parser.parse_args()

        root_folder = None if args.output_folder == '.' else args.output_folder

        return args.expt_name, args.force_download == 'yes', root_folder


    name, force, folder = get_args()
    main(expt_name=name, force_download=force, output_folder=folder)
