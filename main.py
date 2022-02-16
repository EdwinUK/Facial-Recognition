import cv2
import tensorflow as tf

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.logger import Logger
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

from face_detection import FaceDetection
from face_recognition import FaceRecognition
from spoof_detection import SpoofDetection


def preprocess_image(image):
    img = tf.convert_to_tensor(image, "uint8")
    img = tf.image.resize(img, (224, 224))
    return img


class MobileApp(App):
    # Calling the parent class constructor and creating some app widget attributes
    def __init__(self):
        super().__init__()
        self.app_title = Label(text="Facial Recognition", size_hint=(1, .1), font_size='20sp', bold=True)
        self.webcam = Image(size_hint=(1, .7))
        self.button = Button(text="Verify!", on_press=self.verify_user, size_hint=(1, .1), color=[0, 0, 0, 1],
                             background_color=[255, 255, 255, 1])
        self.verified_status = Label(text="Verification waiting to start!", size_hint=(1, .1))
        self.capture = cv2.VideoCapture(0)

    # Building the app
    def build(self):
        # Creating a box layout and adding the widget attributes to it
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.app_title)
        layout.add_widget(self.webcam)
        layout.add_widget(self.verified_status)
        layout.add_widget(self.button)
        Window.size = (250, 450)

        # Schedule the update_capture function to run continuously
        Clock.schedule_interval(self.update_capture, 1.0 / 33.0)

        return layout

    # Updating the webcam feed continuously
    def update_capture(self, *args):
        # Capture each frame and set the frame size to 250x250
        __, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        # Flip the frame horizontal and convert the image to a texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.webcam.texture = img_texture

    # The function which calls of the main classes to perform face detection, spoof detection and face recognition
    def verify_user(self, *args):
        face_detection = FaceDetection()
        __, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        amount_of_faces = face_detection.face_detector(frame)

        if amount_of_faces == 1:
            input_image = preprocess_image(frame)
            # Call the spoof detector method which will detect whether the input is a spoof attack
            spoof_detection = SpoofDetection()
            spoof_result = spoof_detection.spoof_detector(input_image)

            # If the spoof_result is 1 then it's real input, if it's 0 then it's a spoof attack
            if spoof_result == 1:
                # Create an instance of the face recognition class and call the verification function
                # If the 0.5 verification threshold is met then print that the user has been verified
                face_recognition = FaceRecognition()
                verified = face_recognition.face_verification(input_image)
                self.verified_status.text = "You have been verified!" if verified == 1 else "You have not been " \
                                                                                            "recognised! "
                self.verified_status.color = "green" if verified == 1 else "red"
            else:
                self.verified_status.text = "Spoof attack detected!"
                self.verified_status.color = "red"
        else:
            self.verified_status.text = amount_of_faces
            self.verified_status.color = "red"


if __name__ == "__main__":
    MobileApp().run()
