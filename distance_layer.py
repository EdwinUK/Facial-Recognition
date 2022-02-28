from tensorflow.keras import layers
import tensorflow.keras.backend as K


class DistanceLayer(layers.Layer):
    # This layer is responsible for computing the distance
    # between the embeddings

    # Call the parent class constructor
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # This function is used to calculate the euclidean distance by finding the square root of the sum of the squares of
    # the difference between both of the embeddings
    def call(self, anchor, compare):
        sum_squared = K.sum(K.square(anchor - compare), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))
