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

    def face_detector(self):
        while True:
            # Capture each frame
            __, frame = self.capture.read()

            # Store the frame with the detected faces
            result = self.mtcnn.detect_faces(frame)
            if result:
                for person in result:
                    # Store the location of the box and key points
                    bounding_box = person['box']
                    landmarks = person['keypoints']

                    # Slice and store the image to only include the ROI
                    roi_cropped = frame[int(bounding_box[1]):
                                        int(bounding_box[1] + bounding_box[3]),
                                  int(bounding_box[0]):
                                  int(bounding_box[0] + bounding_box[2])]

                    # Display a window showing the cropped face
                    cv2.imshow("ROI", roi_cropped)

                    # if s is pressed then resize the cropped ROI and save it locally
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        roi_cropped = cv2.resize(roi_cropped, (250, 250))
                        cv2.imwrite(f"{self.face_image_path}/{uuid.uuid1()}.jpg", roi_cropped)
                        print("ROI image captured")

                    # Draw a bounding box around the face
                    cv2.rectangle(frame,
                                  (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                  (204, 51, 255),
                                  2)

                    # Draw circles around facial landmarks
                    cv2.circle(frame, (landmarks['left_eye']), 2, (0, 255, 0), 2)
                    cv2.circle(frame, (landmarks['right_eye']), 2, (0, 255, 0), 2)
                    cv2.circle(frame, (landmarks['nose']), 2, (0, 255, 0), 2)
                    cv2.circle(frame, (landmarks['mouth_left']), 2, (0, 255, 0), 2)
                    cv2.circle(frame, (landmarks['mouth_right']), 2, (0, 255, 0), 2)

            # Display the frame with the detected faces
            cv2.imshow('frame', frame)

            # If q is pressed stop the capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the use of the capture and destroy the window
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    # face_detection = FaceDetection()
    # face_detection.face_detector()
    face_recognition = FaceRecognition()
    face_recognition.siamese_model.summary()


if __name__ == "__main__":
    main()
