import gradio as gr


class Interface:
    def __init__(self):
        pass

    @staticmethod
    def render():
        return gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
