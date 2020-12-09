from typing import Tuple, List
from abc import ABC, abstractmethod
import tensorflow as tf

class ModelAssembler(ABC):
    @abstractmethod
    def make_final_layer(self, previous_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        raise NotImplementedError("Your model assembler should override the method make_final_layer")

    @abstractmethod
    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        raise NotImplementedError("Your model assembler should override the method compile_model")

    def make_hidden_layers(self,
                           outputs: List[tf.keras.layers.Layer],
                           network_layers: List[int],
                           dropout_rate: int,
                           activation_fn: str,
                           kernel_initializer: str,
                           regularization_factor: float) -> tf.keras.layers.Layer:
        output_model = tf.keras.layers.Concatenate()(outputs)
        for _, layers in enumerate(network_layers):
            if regularization_factor:
                output_model = tf.keras.layers.Dense(layers,
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                     regularization_factor),
                                                 bias_regularizer=tf.keras.regularizers.l2(
                                                     regularization_factor))(output_model)
            else:
                output_model = tf.keras.layers.Dense(layers, 
                                                     kernel_initializer=kernel_initializer)(output_model)
            output_model = tf.keras.layers.Activation(activation_fn)(output_model)
        if dropout_rate:
            output_model = tf.keras.layers.AlphaDropout(dropout_rate)(output_model)
        return output_model

class BinaryClassificationAssembler(ModelAssembler):
    def __init__(self, output_bias: float):
        self.output_bias = output_bias
    
    def make_final_layer(self, previous_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if self.output_bias is not None:
            output_bias = tf.keras.initializers.Constant(self.output_bias)
        else:
            output_bias = 0
        output_model = tf.keras.layers.Dense(1, bias_initializer=output_bias)(previous_layer)
        output_model = tf.keras.layers.Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model: tf.keras.Model, loss_fn: str,
                      optimizer_fn: str, metrics: List[str]) -> tf.keras.Model:
        model.compile(loss=loss_fn, optimizer=optimizer_fn, metrics=metrics)
        return model
    
class MulticlassClassificationAssembler(ModelAssembler):
    def __init__(self, n_unique_classes: int):
        self.n_unique_classes = n_unique_classes

    def make_final_layer(self, previous_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        output_model = tf.keras.layers.Dense(self.n_unique_classes)(previous_layer)
        output_model = tf.keras.layers.Activation('softmax')(output_model)
        return output_model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

class RegressionClassificationAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        output_model = tf.keras.layers.Dense(1)(previous_layer)
        output_model = tf.keras.layers.Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model