"""
src/model/builder.py
====================
Xây dựng mạng ResNet 1D (Custom) và mạng MLP phục vụ thí nghiệm.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def resblock_1d(x, kernel_size, filters, strides=1):
    """Resblock 1D native với padding='same' để fix lỗi Negative Dimension"""
    x1 = layers.Conv1D(filters, kernel_size, strides=strides, padding="same")(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    
    x1 = layers.Conv1D(filters, kernel_size, padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    
    if strides != 1:
        x = layers.Conv1D(filters, 1, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)

    x1 = layers.Add()([x, x1])
    x1 = layers.ReLU()(x1)
    return x1

class ModelBuilder:
    """Xây dựng ResNet1D cho HASC dataset theo chuẩn cấu hình AutoCPD"""
    def __init__(self, n: int, n_trans: int, kernel_size: int, n_filter: int, # NOTE: kernel_size now is an int
                 dropout_rate: float, n_classes: int, n_resblock: int,
                 m: list, l: int, model_name: str = "ResNet1D"):
        self.n = n
        self.n_trans = n_trans
        
        # Mặc định autocpd dùng kernel 2D (2, 25), nhưng ta chuyển sang 1D nên lấy chiều rộng là 25
        self.kernel_size = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        
        self.n_filter = n_filter
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.n_resblock = n_resblock
        self.m = m
        self.l = l
        self.model_name = model_name

    def build(self) -> tf.keras.Model:
        # Input shape: (Sequence Length = n, Channels = n_trans)
        input_layer = layers.Input(shape=(self.n, self.n_trans), name="Input")
        
        x = layers.Conv1D(self.n_filter, 2, padding="same")(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Fixing padding Negative Dimension issue here
        x = layers.MaxPooling1D(2, padding="same")(x)
        
        j1 = self.n_resblock % 4
        for _ in range(j1):
            x = resblock_1d(x, self.kernel_size, filters=self.n_filter)
            
        j2 = self.n_resblock // 4
        if j2 > 0:
            for _ in range(j2):
                x = resblock_1d(x, self.kernel_size, filters=self.n_filter, strides=2)
                x = resblock_1d(x, self.kernel_size, filters=self.n_filter)
                x = resblock_1d(x, self.kernel_size, filters=self.n_filter)
                x = resblock_1d(x, self.kernel_size, filters=self.n_filter)

        x = layers.GlobalAveragePooling1D()(x)
        
        for i in range(self.l - 1):
            x = layers.Dense(self.m[i], activation="relu", kernel_regularizer="l2")(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
        x = layers.Dense(self.m[self.l - 1], activation="relu", kernel_regularizer="l2")(x)
        output_layer = layers.Dense(self.n_classes, activation="softmax")(x)
        
        model = models.Model(input_layer, output_layer, name=self.model_name)
        return model

class MLPBuilder:
    """
    Xây dựng mạng Dense/MLP đơn giản cho các bài toán phân phối Synthetic (Cauchy, S1, S2, S3)
    Kiến trúc mạng H_{layers, m} theo bài báo.
    """
    def __init__(self, n: int, n_trans: int, n_layers: int, m_neurons: int, 
                 dropout_rate: float=0.2, n_classes: int=2, model_name: str="MLP"):
        self.n = n
        self.n_trans = n_trans
        self.n_layers = n_layers
        self.m_neurons = m_neurons
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.model_name = model_name

    def build(self) -> tf.keras.Model:
        input_layer = layers.Input(shape=(self.n, self.n_trans), name="Input")
        x = layers.Flatten()(input_layer)
        
        for _ in range(self.n_layers):
            x = layers.Dense(self.m_neurons, activation="relu", kernel_regularizer="l2")(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
        output_layer = layers.Dense(self.n_classes, activation="softmax")(x)
        model = models.Model(input_layer, output_layer, name=self.model_name)
        return model
