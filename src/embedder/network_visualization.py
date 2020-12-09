import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pickle
import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from embedder.network_config import *

TITLE_FORMAT = 'Weights for %s'
SCATTER_EMBEDDINGS_FORMAT = '%s_embedding.%s'
PLOT_LOSS_FORMAT = 'loss_epochs.%s'

def make_plot_from_history(history: tf.keras.callbacks.History,
                           output_path: str = None,
                           extension: str = 'pdf') -> Figure:
    """
    Used to make a Figure object containing the loss curve between the epochs.
    :param history: the history outputted from the model.fit method
    :param output_path: (optional) where the image will be saved
    :param extension: (optional) the extension of the file
    :return: a Figure object containing the plot
    """
    loss = history.history['loss']

    fig = plt.figure(figsize=(10, 10))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(loss)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, PLOT_LOSS_FORMAT % extension))

    return fig

def make_visualizations_from_config(config: NeuralEmbedder, extension: str = 'pdf') -> List[Figure]:
    with open(config.get_labels_path(), 'rb') as f:
        labels = pickle.load(f)

    with open(config.get_weights_path(), 'rb') as f:
        embeddings = pickle.load(f)

    n_numerical_cols = len(config.numerical_categories)
    return make_visualizations(labels, embeddings, n_numerical_cols,
                               config.df, config.get_visualizations_dir(), extension)

def make_visualizations(labels: List[LabelEncoder],
                        embeddings: List[np.array],
                        n_numerical_cols: int,
                        df: pd.DataFrame,
                        output_path: str = None,
                        extension: str = 'pdf') -> List[Figure]:
    """
    Used to generate the embedding visualizations for each categorical variable

    :param labels: a list of the LabelEncoders of each categorical variable
    :param embeddings: a Numpy array containing the weights from the categorical variables
    :param n_numerical_cols: number of numerical columns
    :param df: the dataframe from where the weights were extracted
    :param output_path: (optional) where the visualizations will be saved
    :param extension: (optional) the extension to be used when saving the artifacts
    :return: the list of figures for each categorical variable
    """
    #TODO: Right now the cat columns have to be AFTER the numerial columns
    figures = []
    embedded_df = df.iloc[:, n_numerical_cols:df.shape[1]-1]
    for index in range(embedded_df.shape[1]):
        column = embedded_df.columns[index]

        if is_not_single_embedding(labels[index]):
            labels_column = labels[index]
            embeddings_column = embeddings[index]

            pca = PCA(n_components=2)
            Y = pca.fit_transform(embeddings_column)

            fig = plt.figure(figsize=(10, 10))
            figures.append(fig)
            plt.scatter(-Y[:, 0], -Y[:, 1])
            plt.title(TITLE_FORMAT % column)
            for i, text in enumerate(labels_column.classes_):
                plt.annotate(text, (-Y[i, 0], -Y[i, 1]), xytext=(-20, 10), textcoords='offset points')

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                plt.savefig(os.path.join(output_path, SCATTER_EMBEDDINGS_FORMAT % (column, extension)))

    return figures

def is_not_single_embedding(label: LabelEncoder) -> bool:
    """
    Used to check if there is more than one class in a given LabelEncoder
    :param label: label encoder to be checked
    :return: a boolean if the embedding contains more than one class
    """
    return label.classes_.shape[0] > 1