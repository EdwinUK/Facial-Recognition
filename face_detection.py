from mtcnn import MTCNN
import cv2


class FaceDetection:
    # Initializing the following attributes: MTCNN, video capture and the image path
    def __init__(self):
        self.face_detector = MTCNN()
        self.capture = cv2.VideoCapture(0)
        self.face_image_path = "C:/Users/Edwin/OneDrive - University of Greenwich/Year 3/Final Year " \
                               "Project/Programming/Facial-Recognition/cropped_faces"

    def face_detection(self):
        crop_counter = 0
        while True:
            # Capture each frame
            __, frame = self.capture.read()

            # Store the frame with the detected faces
            result = self.face_detector.detect_faces(frame)
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

                    # if s is pressed then save the cropped ROI
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        cv2.imwrite(self.face_image_path + "/" "face_" + str(crop_counter) + ".jpg", roi_cropped)
                        crop_counter += 1

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
    face_detector = FaceDetection()
    face_detector.face_detection()


if __name__ == "__main__":
    main()
