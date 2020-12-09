#!/usr/bin/env python
import numpy as np
import pandas as pd
import config as cfg
import tensorflow as tf
from embedder import network_config, network_maker, network_utility, network_visualization
from embedder import preprocessor

SAVE_MODEL = True
def main():
    csv_file = tf.keras.utils.get_file('titanic.csv', cfg.TITANIC_URL)
    df = pd.read_csv(csv_file)
    df = prepare_titanic(df)
    
    target_name = 'Survived'
    neg, pos = np.bincount(df[target_name])
    output_bias = np.log([pos/neg])
    
    embedded_categories = network_utility.get_categorial_cols(df, target_name)
    numerical_categories = network_utility.get_numerical_cols(df, target_name)
    
    params = {"df": df,
              "target_name": target_name,
              "target_type": network_config.TargetType.BINARY_CLASSIFICATION,
              "train_ratio": 0.8,
              "embedded_categories": embedded_categories,
              "numerical_categories": numerical_categories,
              "network_layers": ([32]),
              "dropout_rate": 0.1,
              "output_bias": output_bias,
              "epochs": 20,
              "batch_size": 128,
              "verbose": True,
              "artifacts_path": cfg.RESULTS_DIR}
    network = network_config.NeuralEmbedder(**params)

    n_numerical_cols = len(network.numerical_categories)
    X_train, X_val, y_train, y_val, labels, scaler = preprocessor.prepare_network_data(network.df,
                                                                               network.target_name,
                                                                               n_numerical_cols,
                                                                               network.train_ratio)
    
    class_weight = preprocessor.get_class_weights(neg, pos)
    model = network_maker.EmbeddingNetwork(network)
    model.fit(X_train, y_train, X_val, y_val, class_weight=class_weight)
    if SAVE_MODEL:
        model.save_model()
    
    embedded_weights = model.get_embedded_weights()
    
    # Save artifacts
    network.save_weights(embedded_weights)
    network.save_labels(labels)
    network.save_scaler(scaler)

    # Make visualization
    network_visualization.make_visualizations_from_config(network, extension='png')

def prepare_titanic(df):
    # Use mean variable for missing Embarked
    df.Embarked[df.Embarked.isnull()] = 'S'
    
    # Fill missing values for age with median
    ages = df.groupby(['Sex', 'Pclass']).Age
    df.Age = ages.transform(lambda x: x.fillna(x.median()))

    # Fill missing values for fare with median
    fares = df.groupby(['Pclass','Embarked']).Fare
    df.Fare = fares.transform(lambda x: x.fillna(x.median()))
    
    # Rearrange columns
    df = df[['Pclass', 'Age', 'SibSp', 'Parch',
             'Fare', 'Sex', 'Embarked', 'Survived']]
    return df

if __name__ == '__main__':
    main()
