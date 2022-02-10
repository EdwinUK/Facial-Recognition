import os
import cv2

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.logger import Logger
from kivy.graphics.texture import Texture
from kivy.clock import Clock

from face_detection import FaceDetection
from face_recognition import FaceRecognition
from spoof_detection import SpoofDetection


class MobileApp(App):
    # Calling the parent class constructor and creating some app widget attributes
    def __init__(self):
        super().__init__()
        self.webcam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify!", on_press=self.verify_user, size_hint=(1, .1))
        self.verified_status = Label(text="Verification waiting to start!", size_hint=(1, .1))
        self.capture = cv2.VideoCapture(0)

    # Building the app
    def build(self):
        # Creating a box layout and adding the widget attributes to it
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.webcam)
        layout.add_widget(self.verified_status)
        layout.add_widget(self.button)

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

        cv2.imwrite(face_detection.input_image_path, frame)
        amount_of_faces = face_detection.face_detector()

        if amount_of_faces == 1:
            # Call the spoof detector method which will detect whether the input is a spoof attack
            spoof_detection = SpoofDetection()
            spoof_result = spoof_detection.spoof_detector()

            # If the spoof_result is 1 then it's real input, if it's 0 then it's a spoof attack
            if spoof_result == 1:
                # Create an instance of the face recognition class and call the verification function
                # If the 0.5 verification threshold is met then print that the user has been verified
                face_recognition = FaceRecognition()
                verified = face_recognition.face_verification(0.5, 0.5)
                self.verified_status.text = "You have been verified!" if verified else "Access denied!"

            else:
                self.verified_status.text = "Spoof attack detected, access denied!"
        else:
            self.verified_status.text = amount_of_faces


if __name__ == "__main__":
    MobileApp().run()
