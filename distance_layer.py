import tensorflow as tf


# Similarity calculation call
def call(self, input_embedding, validation_embedding):
    return tf.math.abs(input_embedding - validation_embedding)


class L1Distance(tf.keras.layers.Layer):
    # Init method for inheritance
    def __init__(self, **kwargs):
        super().__init__()
