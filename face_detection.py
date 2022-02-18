import os
import tensorflow as tf
from mtcnn import MTCNN

# Enable GPU memory growth for tensorflow
physical_devices = tf.config.experimental.list_physical_devices("GPU")
print("Number of GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class FaceDetection:
    # Initializing the following attributes: MTCNN, video capture and paths
    def __init__(self):
        self.mtcnn = MTCNN()

    def face_detector(self, input_image):
        # Read the input image and detect all faces using MTCNN
        faces = self.mtcnn.detect_faces(input_image)

        # Check how many faces are detected in the frame and return the appropriate response
        if len(faces) == 1:
            return 1
        elif len(faces) > 1:
            return "Too many faces are in the frame"
        else:
            return "No face detected in the frame"
