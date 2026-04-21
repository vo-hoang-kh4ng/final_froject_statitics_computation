"""
src/model/builder.py
====================
Wrap hàm build mô hình từ autocpd thành một class.
"""

from autocpd.neuralnetwork import general_deep_nn
import tensorflow as tf

class ModelBuilder:
    def __init__(self, n: int, n_trans: int, kernel_size: tuple, n_filter: int,
                 dropout_rate: float, n_classes: int, n_resblock: int,
                 m: list, l: int, model_name: str):
        self.n = n
        self.n_trans = n_trans
        self.kernel_size = kernel_size
        self.n_filter = n_filter
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.n_resblock = n_resblock
        self.m = m
        self.l = l
        self.model_name = model_name

    def build(self) -> tf.keras.Model:
        """Xây dựng và trả về model Keras chưa compile."""
        model = general_deep_nn(
            n=self.n,
            n_trans=self.n_trans,
            kernel_size=self.kernel_size,
            n_filter=self.n_filter,
            dropout_rate=self.dropout_rate,
            n_classes=self.n_classes,
            n_resblock=self.n_resblock,
            m=self.m,
            l=self.l,
            model_name=self.model_name,
        )
        return model
