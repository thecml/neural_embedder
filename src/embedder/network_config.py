import os
from typing import List
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from embedder.model_assembler import *
from embedder.target_processor import *
from embedder.network_category import *

class TargetType:
    REGRESSION = 0
    BINARY_CLASSIFICATION = 1
    MULTICLASS_CLASSIFICATION = 2

class NeuralEmbedder:
    """
    Used to store all the configuration for TensorFlow network
    """
    def __init__(self,
                 df: pd.DataFrame,
                 target_name: str,
                 target_type: TargetType,
                 train_ratio: float,
                 embedded_categories: List[NetworkCategory],
                 numerical_categories: List[str],
                 network_layers: List[int] = (32, 32),
                 dropout_rate: int = 0,
                 output_bias: float = 0,
                 activation_fn: str = "relu",
                 kernel_initializer: str = 'glorot_uniform',
                 regularization_factor: float = 0,
                 loss_fn: str ='binary_crossentropy',
                 optimizer_fn: str = 'Adam',
                 metrics: List[str] = ['accuracy'],
                 epochs: int = 10,
                 batch_size: int = 128,
                 verbose: bool = False,
                 artifacts_path: str = 'artifacts'):
        
        self.check_not_empty_dataframe(df)
        self.check_target_name(target_name)
        self.check_train_ratio(train_ratio)
        self.check_epochs(epochs)
        self.check_batch_size(batch_size)
        
        self.df = df
        self.target_name = target_name
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.network_layers = network_layers
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.kernel_initializer = kernel_initializer
        self.regularization_factor = regularization_factor
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.metrics = metrics        
        self.verbose = verbose
        self.artifacts_path = artifacts_path
        self.target_processor = self.get_target_processor(target_type)
        self.model_assembler = self.get_model_assembler(target_type, output_bias)
        self.unique_classes = self.df[self.target_name].nunique()
        self.embedded_categories = embedded_categories
        self.numerical_categories = numerical_categories
        self.DEFAULT_WEIGHTS_FILENAME = 'weights.pkl'
        self.DEFAULT_LABELS_FILENAME = 'labels.pkl'
        self.DEFAULT_SCALER_FILENAME = 'scaler.pkl'
        self.DEFAULT_PATH_VISUALIZATIONS = 'visualizations'

    def get_weights_path(self):
        """
        Used to return the path of the stored weights
        :return: the pah of the stored weights on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_WEIGHTS_FILENAME)

    def get_labels_path(self):
        """
        Used to return the path of the stored labels
        :return: the pah of the stored labels on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_LABELS_FILENAME)
    
    def get_scaler_path(self):
        """
        Used to return the path of the stored scaler
        :return: the pah of the stored scaler on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_SCALER_FILENAME)

    def get_visualizations_dir(self):
        """
        Used to return the path of the stored visualizations
        :return: the pah of the stored visualizations on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_PATH_VISUALIZATIONS)
        
    def save_weights(self, weights: List) -> None:
        with open(self.get_weights_path(), 'wb') as f:
            pickle.dump(weights, f, -1)
            
    def save_labels(self, labels: List) -> None:
        with open(self.get_labels_path(), 'wb') as f:
            pickle.dump(labels, f, -1)
            
    def save_scaler(self, scaler: StandardScaler) -> None:
        with open(self.get_scaler_path(), 'wb') as f:
            pickle.dump(scaler, f, -1)

    def check_not_empty_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("You should provide a non-empty pandas dataframe")

    def check_target_name(self, target_name: str) -> None:
        if not target_name:
            raise ValueError("You should provide a non-empty target name")

    def check_target_existent_in_df(self, target_name: str, df: pd.DataFrame) -> None:
        if target_name not in df.columns:
            raise ValueError("You should provide a target variable that is existent on the dataframe")

    def check_train_ratio(self, train_ratio: float) -> None:
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError("You should provide a train ratio greater than 0 and smaller than 1")

    def check_epochs(self, epochs: int) -> None:
        if epochs <= 0:
            raise ValueError("You should provide a epoch greater than zero")

    def check_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("You should provide a batch size greater than zero")
    
    def check_target_type(self, target_type: TargetType) -> None:
        if not isinstance(target_type, TargetType):
            raise ValueError("You should provide a valid target type")
    
    def check_target_processor(self, processor: TargetProcessor) -> None:
        if not isinstance(processor, TargetProcessor):
            raise ValueError("You should provide a target processor that inherits from TargetProcessor")

    def check_model_assembler(self, assembler: ModelAssembler) -> None:
        if not isinstance(assembler, ModelAssembler):
            raise ValueError("You should provide a model assembler that inherits from ModelAssembler")
    
    def get_model_assembler(self,
                            target_type: TargetType,
                            output_bias: float):
        if target_type == TargetType.BINARY_CLASSIFICATION:
            return BinaryClassificationAssembler(output_bias)
        raise ValueError("You should provide a valid assembler type")

    def get_target_processor(self, type: int) -> TargetProcessor:
        if type == TargetType.BINARY_CLASSIFICATION:
            return BinaryClassificationProcessor()
        raise ValueError("You should provide a valid target type")