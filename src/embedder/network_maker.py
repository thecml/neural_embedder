import numpy as np
from typing import List, Tuple
import tensorflow as tf
from embedder.network_config import *
from embedder.network_utility import *

class EmbeddingNetwork:
    """
    This class is used to provide a Entity Embedding Network from a given EmbeddingConfig object
    """
    def __init__(self, config: NeuralEmbedder):
        super().__init__()
        self.config = config
        self.model = self.__make_model()

    def __make_model(self) -> tf.keras.Model:
        """
        This method is used to generate our Model containing the Embedding layers alongside with the output layers
        :return: a compiled Model object
        """
        dense_inputs, dense_outputs = self._make_numerical_layers(self.config.kernel_initializer,
                                                              self.config.activation_fn)
        embedded_inputs, embedded_outputs = self._make_embedding_layers()
        
        model_inputs = dense_inputs + embedded_inputs
        model_outputs = dense_outputs + embedded_outputs
        
        output_model = self.config.model_assembler.make_hidden_layers(model_outputs,
                                                                      self.config.network_layers,
                                                                      self.config.dropout_rate,
                                                                      self.config.activation_fn,
                                                                      self.config.kernel_initializer,
                                                                      self.config.regularization_factor)
        output_model = self.config.model_assembler.make_final_layer(output_model)

        model = tf.keras.Model(inputs=model_inputs, outputs=output_model)
        model = self.config.model_assembler.compile_model(model,
                                                          self.config.loss_fn,
                                                          self.config.optimizer_fn,
                                                          self.config.metrics)
        return model

    def _make_numerical_layers(self, kernel_initializer: str,
                           activation_fn: str) -> Tuple[List[tf.keras.layers.Layer],
                                                        List[tf.keras.layers.Layer]]:
        numerical_inputs = []
        numerical_outputs = []
        
        for category in self.config.numerical_categories:
            input_category = tf.keras.layers.Input(shape=(1,))
            output_category = tf.keras.layers.Dense(1, name=category,
                                                    kernel_initializer=kernel_initializer,
                                                    activation=activation_fn)(input_category)
            
            numerical_inputs.append(input_category)
            numerical_outputs.append(output_category)
        
        return numerical_inputs, numerical_outputs
        
    def _make_embedding_layers(self) -> Tuple[List[tf.keras.layers.Layer], List[tf.keras.layers.Layer]]:
        """
        This method is used to generate the list of inputs and output layers where our Embedding layers will be placed
        :return: a tuple containing two lists: the first, the input layers; the second, the output layers
        """
        embedding_inputs = []
        embedding_outputs = []

        for category in self.config.embedded_categories:
            input_category = tf.keras.layers.Input(shape=(1,))
            output_category = tf.keras.layers.Embedding(input_dim=category.unique_values,
                                        output_dim=category.embedding_size,
                                        name=category.alias)(input_category)
            output_category = tf.keras.layers.Reshape(target_shape=(category.embedding_size,))(output_category)

            embedding_inputs.append(input_category)
            embedding_outputs.append(output_category)

        return embedding_inputs, embedding_outputs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray, y_valid: np.ndarray, class_weight: dict = None) -> tf.keras.callbacks.History:
        """
        This method is used to fit a given training and validation data into our entity embeddings model
        :param X_train: training features
        :param y_train: training targets
        :param X_valid: validation features
        :param y_valid: validation targets
        :return a History object
        """
        # Use for regression
        #self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
        history = self.model.fit(x=transpose_to_list(X_train),
                                 y=y_train,
                                 validation_data=(transpose_to_list(X_valid), y_valid),
                                 epochs=self.config.epochs,
                                 batch_size=self.config.batch_size,
                                 class_weight=class_weight)
        return history

    def save_model(self) -> None:
        self.model.save(self.config.artifacts_path)

    def get_embedded_weights(self) -> List:
        weights_embeddings = []
        columns_to_skip = self.config.numerical_categories + [self.config.target_name]
        for column in get_all_columns_except(self.config.df, columns_to_skip):
            weights = self._get_weights_from_layer(column)
            weights_embeddings.append(weights)

        return weights_embeddings

    def _get_weights_from_layer(self, layer_name: str):
        return self.model.get_layer(layer_name).get_weights()[0]

    def _val_for_fit(self, val):
        val = np.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return np.exp(val * self.max_log_y)