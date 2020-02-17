from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from pprint import pprint
from copy import copy
import random
import cv2
import numpy
from random import randint
from cv2utils import array_from_video, array_to_video

import matplotlib.pyplot as plt


VIDEO_FILE = "test.avi"

vid_arr = array_from_video(VIDEO_FILE)
"""import pickle to nima veze z diskom... kivy je problem
with open("temp.pkl", "wb") as f:
    pickle.dump(vid_arr_, f)
with open("temp.pkl", "rb") as f:
    vid_arr = pickle.load(f)"""


class Screen(Widget):

    def __init__(self, slider: Slider, **kwargs):
        super(Screen, self).__init__(**kwargs)
        # self.redraw()
        self.slider = slider
        self.current_frame = 0
        self.prev_touch = None
        self.paint_mode = False
        self.active_region = None

    def redraw(self):
        texture = Texture.create(size=self.size, colorfmt="rgb")
        size = tuple([int(x) for x in self.size])
        arr = vid_arr[self.current_frame]  # [:,:,0]
        arr = cv2.resize(arr, tuple(size), interpolation=cv2.INTER_LINEAR)
        data = arr.tostring()
        data = bytes(reversed(data))
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        self.slider.value = int(1000 * self.current_frame / vid_arr.shape[0])
        with self.canvas:
            self.canvas.clear()
            Rectangle(texture=texture, pos=self.pos, size=self.size)
        # self.bg_rect.size = self.size
        # self.bg_rect.pos = self.pos
        # self.bind(pos=10, size=10)

    def on_touch_down(self, touch):
        pass

    def on_touch_move(self, touch):
        if not self.paint_mode:
            if self.prev_touch is not None:
                if self.prev_touch.x > touch.x:
                    self.current_frame -= 1
                else:
                    self.current_frame += 1
                self.current_frame = max(self.current_frame, 0)
                self.current_frame = min(self.current_frame, vid_arr.shape[0] - 1)
            self.prev_touch = copy(touch)
        if self.paint_mode:
            center = (int(vid_arr.shape[2] - touch.x * vid_arr.shape[2] / self.size[0]),
                      int(vid_arr.shape[1] - touch.y * vid_arr.shape[1] / self.size[1]))
            # print(f"center: {center}")

            vid_arr[self.current_frame] = cv2.circle(vid_arr[self.current_frame], center=center, radius=5,
                                                     color=(0, 255, 0), thickness=-1)
        if self.active_region is not None:
            #pprint(vid_arr[self.current_frame][self.active_region])
            vid_arr[self.current_frame][self.active_region] = (0, 255, 0)
        self.redraw()
        # print(f"{self.current_frame}/{vid_arr.shape[0]}")
        # print(f"paint mode {self.paint_mode}")
        # Rectangle(texture=texture, pos=self.pos, size=(vid_arr.shape[1], vid_arr.shape[2]))

    def toggle_paint_mode(self):
        self.paint_mode = not self.paint_mode

    def select_region(self):
        if self.active_region is None:
            self.active_region = [vid_arr[self.current_frame] == (0, 255, 0)]
            self.active_region = numpy.array(self.active_region)[0,:,:,0]* numpy.array(self.active_region)[0,:,:,1] * numpy.array(self.active_region)[0,:,:,2]
            #plt.imshow(numpy.array(self.active_region)[0,:,:,0]* numpy.array(self.active_region)[0,:,:,1] * numpy.array(self.active_region)[0,:,:,2])
            #plt.show()
        else:
            self.active_region = None

    def save(self):
        array_to_video(vid_arr)

    #def showtime(self):
        #print("lol%s" % self.current_frame)


class PropagateButton(Button):
    def __init__(self, screen: Screen, **kwargs):
        super(PropagateButton, self).__init__(**kwargs)
        self.text = "Propagate"
        self.size_hint = (None, 0.1)
        self.screen = screen

    def on_press(self):
        self.screen.select_region()
        if self.screen.active_region is None:
            self.background_color = (1, 0, 0, 1)
        else:
            self.background_color = (0, 1, 0, 1)

class SaveButton(Button):
    def __init__(self, screen: Screen, **kwargs):
        super(SaveButton, self).__init__(**kwargs)
        self.text = "Save"
        self.size_hint = (None, 0.1)
        self.screen = screen

    def on_press(self):
        self.screen.save()


class PaintModeButton(Button):
    def __init__(self, screen: Screen, **kwargs):
        super(PaintModeButton, self).__init__(**kwargs)
        self.text = "Paint mode"
        self.size_hint = (None, 0.1)
        self.screen = screen
        self.background_color = (1, 0, 0, 1)

    def on_press(self):
        #self.screen.showtime()
        self.screen.toggle_paint_mode()
        if self.screen.paint_mode == False:
            self.background_color = (1, 0, 0, 1)
        if self.screen.paint_mode == True:
            self.background_color = (0, 1, 0, 1)

#class


class MyApp(App):

    def build(self):
        # a  = LoginScreen()
        # Clock.schedule_interval(a.update, 1.0 / 60.0)
        # layout = GridLayout(cols=2)
        layout = StackLayout()
        slider = Slider(min=0, max=1000, value=0, size_hint=(0.5, 0.1))
        showtime = Screen(slider=slider, size_hint=(1., 0.9))

        # layout.add_widget(Button(text="suckamidick"))
        layout.add_widget(SaveButton(screen=showtime))
        layout.add_widget(PropagateButton(screen=showtime))
        layout.add_widget(PaintModeButton(screen=showtime))
        layout.add_widget(slider)
        layout.add_widget(showtime)

        return layout


if __name__ == '__main__':
    MyApp().run()

"""class LoginScreen(GridLayout):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        self.a = Label(text="User Name")
        self.cols = 2
        self.rows = 2
        self.add_widget(Screen())
        self.add_widget(Button(text='Hello 1'))
        self.add_widget(Button(text='Hello 1'))
        self.add_widget(Button(text='Hello 1'))
        # self.username = TextInput(multiline=False)
        # self.add_widget(self.username)
        # self.add_widget(Label(text='password'))
        # self.password = TextInput(password=True, multiline=False)
        # self.add_widget(self.password)
        self.i = 0

    def on_touch_move(self, touch):
        self.a.text = "lolzies"

    def update(self,dt):
        self.i+=1
        #print(self.i)
        if self.i > 500:
            self.a.text = "lolzies"""

"""if randint(0,10)<5:
    with self.canvas:
        Rectangle(source="phase.jpg", pos=self.pos, size=self.size)
else:
    with self.canvas:
        Rectangle(source="gray_im.jpg", pos=self.pos, size=self.size)"""
