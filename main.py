import os
import cv2
import cvzone
import uuid

from kivy.app import App
from kivy.properties import BooleanProperty
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.recycleview import RecycleView

from face_detection import FaceDetection
from face_recognition import FaceRecognition
from spoof_detection import SpoofDetection

# Global variable to pass the index of a name between classes when removing a user from the database
global name_index


# (Selectable Layout class from Kivy) Adds focus and selection behaviour to the layout
class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    pass


# (Selectable label class from Kivy) Adds selection support to label for the recycle view layout
class SelectableLabel(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    # Catch and handle the view changes
    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    # Add selection on touch down
    def on_touch_down(self, touch):
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    # Respond to the selection of items in the view
    def apply_selection(self, rv, index, is_selected):
        self.selected = is_selected
        if is_selected:
            global name_index
            name_index = index


# (Recycle view class from Kivy) The init function calls the super constructor and data is where the database names
# will be stored
class RV(RecycleView):
    def __init__(self, **kwargs):
        super(RV, self).__init__(**kwargs)
        self.data = [{"text": str(name).split(".")[0]} for name in os.listdir("face_db")]


class MyFaceApp(App):
    # Calling the parent class constructor and creating some class attributes
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.webcam = Image(size_hint=(1, .7))
        self.status = Label(text="Verification or registration is ready to begin!", font_size='18sp',
                            size_hint=(1, .1), font_name="Arial", color=[1, 1, 1, 1])

        self.new_users_name = TextInput(text="", size_hint=(1, .12), halign="center", font_name="Arial",
                                        cursor_color=[0, 0, 0, 1], multiline=False)
        self.temp_file = ""
        self.register_popup_instance = None
        self.remove_popup_instance = None
        self.list_view = None

    # Building the main app
    def build(self):
        # Creating some widget attributes that have longer parameters
        app_title = Label(text="Facial Recognition", size_hint=(1, .1), font_size='24sp',
                          font_name="Arial")

        verify_button = Button(text="Verify an existing face!", on_press=self.verify_user, size_hint=(.6, .1),
                               pos_hint={'x': .2, 'y': .2}, font_size='22sp', color=[0, 0, 0, 1],
                               background_normal="button.png", font_name="Arial")

        register_button = Button(text="Register a new face!", on_press=self.register_user, size_hint=(.6, .1),
                                 pos_hint={'x': .2, 'y': .2}, font_size='22sp',
                                 color=[0, 0, 0, 1], background_normal="button.png", font_name="Arial")

        remove_button = Button(text="Remove an existing face!", on_press=self.build_remove_popup, size_hint=(.6, .1),
                               pos_hint={'x': .2, 'y': .2}, font_size='22sp',
                               color=[0, 0, 0, 1], background_normal="button.png", font_name="Arial")

        # Creating a box layout and more widgets, then adding the widgets to the layout
        main_screen = BoxLayout(orientation="vertical")
        main_screen.add_widget(app_title)
        main_screen.add_widget(self.webcam)
        main_screen.add_widget(self.status)
        main_screen.add_widget(Label(text="", size_hint=(1, .05)))
        main_screen.add_widget(verify_button)
        main_screen.add_widget(Label(text="", size_hint=(1, .025)))
        main_screen.add_widget(register_button)
        main_screen.add_widget(Label(text="", size_hint=(1, .025)))
        main_screen.add_widget(remove_button)
        main_screen.add_widget(Label(text="", size_hint=(1, .05)))

        # Setting a fixed size for the window and making the background colour grey
        Window.size = (700, 675)
        Window.clearcolor = (30 / 255.0, 30 / 255.0, 30 / 255.0, 1)

        # Schedule the update_capture function to run continuously
        Clock.schedule_interval(self.update_capture, 1.0 / 33.0)

        return main_screen

    # Constructing the register process popup by creating and adding widgets to a box layout which is then used as
    # content for the popup
    def build_register_popup(self, *args):
        if self.register_popup_instance is None:

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

            self.register_popup_instance = register_popup
        else:
            self.register_popup_instance.open()

    # Constructing the user remove process popup by creating and adding widgets to a box layout which is then used as
    # content for the popup
    def build_remove_popup(self, *args):
        self.list_view = RV()

        remove_popup_content = BoxLayout(orientation="vertical")

        remove_popup_content.add_widget(Label(text="", size_hint=(1, .1)))

        remove_popup_content.add_widget(self.list_view)

        remove_popup_content.add_widget(Label(text="", size_hint=(1, .1)))

        remove_popup_content.add_widget(Button(text="Remove user!", on_press=self.remove_user,
                                               size_hint=(.5, .25), pos_hint={'x': .25, 'y': .2},
                                               font_name="Arial", background_normal="button.png",
                                               color=[0, 0, 0, 1]))

        remove_popup_content.add_widget(Label(text="", size_hint=(1, .1)))

        remove_face_popup = Popup(title='Face Removal Process', content=remove_popup_content,
                                  size_hint=(.7, .5), title_size='18sp', title_font="Arial", title_align="center")
        remove_face_popup.open()

        self.remove_popup_instance = remove_face_popup

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

        # If one face is detected then proceed with spoof detection otherwise set the status accordingly
        if len(faces) == 1:
            # Call the spoof detector method which will detect whether the input is a spoof attack
            spoof_detection = SpoofDetection()
            spoof_result = spoof_detection.spoof_detector(frame)

            # If the spoof_result is 1 then it's real input, if it's 0 then it's a spoof attack
            if spoof_result == 1:
                # Create an instance of the face recognition class
                face_recognition = FaceRecognition()

                # Call the verification function passing the face coordinates and the frame
                # if the 0.5 verification threshold is met then show the user's name else print unknown face
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

    # This function is used for registering a new user's face
    def register_user(self, *args):
        # Create and instance of the face detection class and call the face_detector method returning a frame and faces
        face_detection = FaceDetection()
        faces, frame = face_detection.face_detector(self.capture)

        # If one face is detected then proceed with registration otherwise set the status accordingly
        if len(faces) == 1:
            # Crop and store ROI from the frame
            bounding_box = faces[0]["box"]
            frame = frame[int(bounding_box[1]):
                          int(bounding_box[1] + bounding_box[3]),
                          int(bounding_box[0]):
                          int(bounding_box[0] + bounding_box[2])]

            # Create a temporary filename and save this ROI under that temporary name whilst calling the popup
            self.temp_file = uuid.uuid1()
            cv2.imwrite(f"face_db/{self.temp_file}.jpg", frame)
            self.build_register_popup()
        else:
            self.status.text = faces
            self.status.color = "red"

    # This function will change the name of the temporary file to the new user's name which they submitted
    def successful_registration(self, *args):
        os.rename(f"face_db/{self.temp_file}.jpg", f"face_db/{self.new_users_name.text}.jpg")
        self.status.text = f"A new face has been successfully registered!"
        self.status.color = "green"
        self.temp_file = ""
        self.register_popup_instance.dismiss()

    # If the user does not complete registration by submitting a name for the file then the file will be deleted
    def unsuccessful_registration(self, *args):
        if self.temp_file != "":
            os.remove(f"face_db/{self.temp_file}.jpg")
            self.status.text = f"The face registration was not successfully completed!"
            self.status.color = "red"
            self.temp_file = ""

    # When the function is called it will use the global variable index to find the name in the data variable
    # then removing this user's face from the database and lastly refreshing the popup
    def remove_user(self, *args):
        file_name = f"{self.list_view.data[name_index]['text']}.jpg"
        os.remove(f"face_db/{file_name}")
        self.remove_popup_instance.dismiss()
        self.build_remove_popup()
        self.status.text = f"The user has been successfully removed from the database!"
        self.status.color = "green"


if __name__ == "__main__":
    MyFaceApp().run()
