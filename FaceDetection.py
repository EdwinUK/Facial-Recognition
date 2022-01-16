from mtcnn import MTCNN
import cv2

# Storing the MTCNN function and the capture device
face_detector = MTCNN()
capture = cv2.VideoCapture(0)


def face_detection():
    while True:
        # Capture each frame
        __, frame = capture.read()

        # Store the frame with the detected faces
        result = face_detector.detect_faces(frame)
        if result:
            for person in result:
                # Store the location of the box and key points
                bounding_box = person['box']
                landmarks = person['keypoints']

                # Create a bounding box around the face
                cv2.rectangle(frame,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (204, 51, 255),
                              2)

                # Create circles around facial landmarks
                cv2.circle(frame, (landmarks['left_eye']), 2, (0, 255, 0), 2)
                cv2.circle(frame, (landmarks['right_eye']), 2, (0, 255, 0), 2)
                cv2.circle(frame, (landmarks['nose']), 2, (0, 255, 0), 2)
                cv2.circle(frame, (landmarks['mouth_left']), 2, (0, 255, 0), 2)
                cv2.circle(frame, (landmarks['mouth_right']), 2, (0, 255, 0), 2)

        # Display the frame with the detected faces
        cv2.imshow('frame', frame)
        # Stop the capture if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the use of the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()


def main():
    face_detection()


if __name__ == "__main__":
    main()
