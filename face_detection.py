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
        self.mtcnn = MTCNN()
        self.capture = cv2.VideoCapture(0)
        self.root_path = os.path.dirname(os.path.abspath("face_detection.py"))
        self.face_image_path = os.path.join(self.root_path, "cropped_faces")
        self.input_image_path = os.path.join(self.root_path, "application_data", "input_image", "input_image.jpg")

    def face_detector(self):
        while True:
            # Capture each frame
            __, frame = self.capture.read()

            # Display the frame with the detected faces
            cv2.imshow('Face Recognition', frame)

            # If q is pressed stop the capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # If s is pressed then resize the cropped ROI and save it locally
            # if cv2.waitKey(1) & 0xFF == ord('s'):
            #     roi_cropped = cv2.resize(roi_cropped, (250, 250))
            #     cv2.imwrite(f"{self.face_image_path}/{uuid.uuid1()}.jpg", roi_cropped)
            #     print("ROI image captured")

            # If v is pressed then save the frame locally and let MTCNN read it
            if cv2.waitKey(10) & 0xFF == ord('v'):
                cv2.imwrite(self.input_image_path, frame)
                pixels = cv2.imread(self.input_image_path)
                result = self.mtcnn.detect_faces(pixels)

                # Only process the image if there is 1 face in the frame
                if len(result) == 1:
                    for person in result:
                        # Store the location of the box and key points
                        bounding_box = person['box']
                        landmarks = person['keypoints']

                        # Slice and store the image to only include the ROI
                        roi_cropped = frame[int(bounding_box[1]):
                                            int(bounding_box[1] + bounding_box[3]),
                                            int(bounding_box[0]):
                                            int(bounding_box[0] + bounding_box[2])]
                        cv2.imwrite(self.input_image_path, roi_cropped)

                        # If the 0.5 verification threshold is met then print that the user has been verified
                        face_recognition = FaceRecognition()
                        results, verified = face_recognition.verification(0.5, 0.9)
                        print("You have been verified!" if verified else "Access denied!")
                else:
                    print("Too many faces in the image")

        # Release the use of the capture and destroy the window
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    face_detection = FaceDetection()
    face_detection.face_detector()


if __name__ == "__main__":
    main()
