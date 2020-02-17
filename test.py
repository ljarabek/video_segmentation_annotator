from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.image import CoreImage
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from PIL import Image

kv = Builder.load_string('''
#:kivy 1.11.0

<RootWidget>:
    img: img
    img3: img3
    img4: img4
    do_default_tab: False

    TabbedPanelItem:
        text: 'PIL Image'

        Screen:
            RelativeLayout:
                Image:
                    id: img
                    pos_hint: {"left": 1, 'bottom': 1}
                    size_hint: 0.5, 1
                    allow_stretch: True

            RelativeLayout:
                Image:
                    id: img3
                    pos_hint: {"right": 1, 'bottom': 1}
                    size_hint: 0.5, 1
                    allow_stretch: True

    TabbedPanelItem:
        text: 'canvas'

        Screen:
            FloatLayout:
                Image:
                    id: img4
                    keep_data: True
                    allow_stretch: True
                    canvas.before:
                        Color:
                            rgba: 0, 0, 0, 1  # Black
                        Rectangle:
                            pos: self.pos
                            size: self.size


''')


class RootWidget(TabbedPanel):

    def __init__(self, **kwargs):
        super(RootWidget, self).__init__(**kwargs)
        iw = Image.open("./20170927-lung-mass.jpg")   # Use PIL.Image
        iw.save('./phase.jpg')
        gray = iw.convert('1')
        gray.save('./gray_im.jpg')
        self.img.source = './phase.jpg'
        self.img3.texture = CoreImage('./gray_im.jpg').texture
        self.img4.source = './gray_im.jpg'




class KivyPILApp(App):
    title = "Kivy & PIL Demo"

    def build(self):
        return RootWidget()


if __name__ == "__main__":
    KivyPILApp().run()