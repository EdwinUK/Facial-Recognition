import os

import cv2
import cvzone
import tensorflow as tf
import uuid

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.core.window import Window

from face_detection import FaceDetection
from face_recognition import FaceRecognition
from spoof_detection import SpoofDetection

# Enable GPU memory growth for tensorflow
physical_devices = tf.config.experimental.list_physical_devices("GPU")
print("Number of GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MyFaceApp(App):
    # Calling the parent class constructor and creating some widget attributes
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.webcam = Image(size_hint=(1, .7))
        self.status = Label(text="Verification or registration is ready to begin!", font_size='18sp',
                            size_hint=(1, .1), font_name="Arial", color=[1, 1, 1, 1])

        self.new_users_name = TextInput(text="", size_hint=(1, .12), halign="center", font_name="Arial",
                                        cursor_color=[0, 0, 0, 1], multiline=False)

        self.temp_file = ""
        self.popup_instance = None

    # Building the app
    def build(self):
        app_title = Label(text="Facial Recognition", size_hint=(1, .1), font_size='24sp',
                          font_name="Arial")

        verify_button = Button(text="Verify an existing face!", on_press=self.verify_user, size_hint=(.6, .1),
                               pos_hint={'x': .2, 'y': .2}, font_size='22sp', color=[0, 0, 0, 1],
                               background_normal="button.png", font_name="Arial")

        register_button = Button(text="Register a new face!", on_press=self.register_user, size_hint=(.6, .1),
                                 pos_hint={'x': .2, 'y': .2}, font_size='22sp',
                                 color=[0, 0, 0, 1], background_normal="button.png", font_name="Arial")

        # Creating a box layout and adding the widget attributes to it
        main_screen = BoxLayout(orientation="vertical")
        main_screen.add_widget(app_title)
        main_screen.add_widget(self.webcam)
        main_screen.add_widget(self.status)
        main_screen.add_widget(Label(text="", size_hint=(1, .05)))
        main_screen.add_widget(verify_button)
        main_screen.add_widget(Label(text="", size_hint=(1, .025)))
        main_screen.add_widget(register_button)
        main_screen.add_widget(Label(text="", size_hint=(1, .05)))

        Window.size = (700, 650)
        Window.clearcolor = (30 / 255.0, 30 / 255.0, 30 / 255.0, 1)

        # Schedule the update_capture function to run continuously
        Clock.schedule_interval(self.update_capture, 1.0 / 33.0)

        return main_screen

    def build_popup(self, *args):
        if self.popup_instance is None:

            popup_content = BoxLayout(orientation="vertical")

            popup_content.add_widget(Label(text="Enter your first and last name with a space in the middle",
                                           size_hint=(1, .3), halign="center", font_name="Arial"))

            popup_content.add_widget(self.new_users_name)

            popup_content.add_widget(Label(text="", size_hint=(1, .1)))

            popup_content.add_widget(Button(text="Save your face!", on_press=self.successful_registration,
                                            size_hint=(.6, .15), pos_hint={'x': .2, 'y': .2},
                                            font_name="Arial", background_normal="button.png", color=[0, 0, 0, 1]))

            popup_content.add_widget(Label(text="", size_hint=(1, .05)))

            register_popup = Popup(title='Face Registering Process', content=popup_content,
                                   size_hint=(.6, .4), title_size='18sp', title_font="Arial", title_align="center",
                                   on_dismiss=self.unsuccessful_registration)
            register_popup.open()

            self.popup_instance = register_popup
        else:
            self.popup_instance.open()

    # Updating the webcam feed continuously
    def update_capture(self, *args):
        # Capture each frame and create an overlay border which will appear on the GUI
        __, original_frame = self.capture.read()
        overlay = cv2.imread("face_border.png", cv2.IMREAD_UNCHANGED)
        overlay = cvzone.overlayPNG(original_frame, overlay, [205, 125])

        # Flip the frame horizontal and convert the image to a texture
        buf = cv2.flip(overlay, 0).tobytes()
        img_texture = Texture.create(size=(overlay.shape[1], overlay.shape[0]), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.webcam.texture = img_texture

    # The function which calls of the main classes to perform face detection, spoof detection and face recognition
    def verify_user(self, *args):
        face_detection = FaceDetection()
        faces, frame = face_detection.face_detector(self.capture)

        if len(faces) == 1:
            # Call the spoof detector method which will detect whether the input is a spoof attack
            spoof_detection = SpoofDetection()
            spoof_result = spoof_detection.spoof_detector(frame)

            # If the spoof_result is 1 then it's real input, if it's 0 then it's a spoof attack
            if spoof_result == 1:
                # Create an instance of the face recognition class
                face_recognition = FaceRecognition()

                # Call the verification function, if the 0.5 verification threshold is met then show user's name
                face_result = face_recognition.face_verification(faces, frame, 0.5)
                self.status.text = f"Recognised as: {face_result}" if not face_result == "Unknown Face!" else \
                    "Unknown Face!"
                self.status.color = "green" if not face_result == "Unknown Face!" else "red"
            else:
                self.status.text = "Spoof attack detected!"
                self.status.color = "red"
        else:
            self.status.text = faces
            self.status.color = "red"

    def register_user(self, *args):
        face_detection = FaceDetection()
        amount_of_faces, frame = face_detection.face_detector(self.capture)

        if len(amount_of_faces) == 1:
            for face in amount_of_faces:
                bounding_box = face['box']
                frame = frame[int(bounding_box[1]):
                              int(bounding_box[1] + bounding_box[3]),
                              int(bounding_box[0]):
                              int(bounding_box[0] + bounding_box[2])]

            self.temp_file = uuid.uuid1()
            cv2.imwrite(f"face_db/{self.temp_file}.jpg", frame)
            self.build_popup()
        else:
            self.status.text = amount_of_faces
            self.status.color = "red"

    def successful_registration(self, *args):
        os.rename(f"face_db/{self.temp_file}.jpg", f"face_db/{self.new_users_name.text}.jpg")
        self.status.text = f"Your face has been successfully registered!"
        self.status.color = "green"
        self.temp_file = ""
        self.popup_instance.dismiss()

    def unsuccessful_registration(self, *args):
        if self.temp_file != "":
            os.remove(f"face_db/{self.temp_file}.jpg")
            self.status.text = f"The face registration was not successfully completed!"
            self.status.color = "red"
            self.temp_file = ""


if __name__ == "__main__":
    MyFaceApp().run()
