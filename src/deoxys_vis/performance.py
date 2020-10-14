# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


"""
This file contains multiple helper function for plotting diagram and images
using matplotlib
"""


import matplotlib.pyplot as plt
from .utils import read_csv


def plot_log_performance_from_csv(filepath, output_path):
    """
    Plot and save multiple performance figure using a log file generated from
    tensorflow.keras.callbacks.CSVLogger

    Parameters
    ----------
    filepath : str
        filename of the log file
    output_path : str
        path to the folder for saving plotted diagram
    """
    df = read_csv(filepath, index_col='epoch')

    # Plot all data
    _plot_data(df, 'All parameters', df.columns, output_path + '/all.png')

    # Plot train data
    train_keys = [key for key in df.columns if 'val' not in key]
    _plot_data(df, 'Train Performance', train_keys, output_path + '/train.png')

    # Plot val data
    val_keys = [key for key in df.columns if 'val' in key]
    _plot_data(df, 'Validation Performance',
               val_keys, output_path + '/val.png')

    # Plot compare train and val
    compare = [[key, 'val_' + key] for key in train_keys]
    for keys in compare:
        _plot_data(df, '{} Performance'.format(
            keys[0]), keys, output_path + '/{}.png'.format(keys[0]))


def plot_evaluation_performance_from_csv(filepath, output_path):
    # Load data to file
    df = read_csv(filepath, index_col='epoch')

    # Plot evaluation
    _plot_data(df, 'Evaluation Performance', df.columns,
               output_path + '/evaluation.png')


def _plot_data(dataframe, name, columns, filename):
    ax = dataframe[columns].plot()
    epoch_num = dataframe.shape[0]
    if epoch_num < 20:
        ax.set_xticks(dataframe.index)
    else:
        tick_distance = epoch_num // 10
        ax.set_xticks([tick for i, tick in enumerate(
            dataframe.index) if i % tick_distance == 0])
    ax.set_title(name)
    plt.savefig(filename)
