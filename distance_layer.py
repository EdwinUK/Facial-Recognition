import tensorflow as tf


class L1Distance(tf.keras.layers.Layer):
    # Init method for inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation call
    @staticmethod
    def call(input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
