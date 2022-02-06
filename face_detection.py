from mtcnn import MTCNN
import cv2
import os
import tensorflow as tf
import uuid

from face_recognition import FaceRecognition

physical_devices = tf.config.experimental.list_physical_devices("GPU")
print("Number of GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class FaceDetection:
    # Initializing the following attributes: MTCNN, video capture and paths
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.root_path = os.path.dirname(os.path.abspath("face_detection.py"))
        self.input_image_path = os.path.join(self.root_path, "application_data", "input_image", "input_image.jpg")

    def face_detector(self):
        while True:
            # Capture each frame and set the frame size to 250x250
            __, frame = self.capture.read()
            frame = frame[120:120 + 250, 200:200 + 250, :]

            # Display the frame with the detected faces
            cv2.imshow('Face Recognition', frame)

            # If q is pressed stop the capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # If v is pressed then save the frame locally
            if cv2.waitKey(10) & 0xFF == ord('v'):
                cv2.imwrite(self.input_image_path, frame)

                # Create an instance of the face recognition class and call the verification function
                # If the 0.5 verification threshold is met then print that the user has been verified
                face_recognition = FaceRecognition()
                results, verified = face_recognition.verification(0.5, 0.5)
                print("You have been verified!" if verified else "Access denied!")

        # Release the use of the capture and destroy the window
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    face_detection = FaceDetection()
    face_detection.face_detector()


if __name__ == "__main__":
    main()
