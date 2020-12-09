import numpy as np

class NetworkCategory:
    """
    Used to store fields related to a given category,
    such as its name, count of unique values and the size of each embedding layer
    """
    def __init__(self, alias: str, unique_values: int):
        self.alias = alias
        self.unique_values = unique_values
        self.embedding_size = self.get_embedding_size(unique_values)
        
    def get_embedding_size(self, unique_values: int) -> int:
        """
        Return the embedding size to be used on the Embedding layer
        :param unique_values: the number of unique values in the given category
        :return: the size to be used on the embedding layer
        """
        size = int(min(np.ceil(unique_values / 2), 50))
        if size < 2:
            return 2
        else:
            return size