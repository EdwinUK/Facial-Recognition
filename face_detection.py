from mtcnn import MTCNN
import cv2


class FaceDetection:
    # Initializing the following attributes: MTCNN, video capture and paths
    def __init__(self):
        self.mtcnn = MTCNN()

    def face_detector(self, capture):
        # Read the input image and detect all faces using MTCNN
        __, frame = capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        faces = self.mtcnn.detect_faces(frame)
        # Check how many faces are detected in the frame and return the appropriate response
        if len(faces) == 1:
            return faces, frame
        elif len(faces) > 1:
            return "Too many faces are in the frame", None
        else:
            return "No face detected in the frame", None
