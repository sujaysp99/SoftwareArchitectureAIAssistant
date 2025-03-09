import gradio as gr
import os

import fitz
from PIL import Image

def createUI():
    with gr.Blocks() as demo:
        with gr.Row():
            api_key = gr.Textbox(
                placeholder='Enter OpenAI API key',
                interactive=True,
                type="password",
                label='API Key'
            )
            change_api_key = gr.Button('Change Key')

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label='Software Architect AI', type='messages', height=680)
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter"
                )
                submit_btn = gr.Button('Submit')
            
            with gr.Column(scale=1):
                pdf_display = gr.Image(label='Uploaded PDF Page', interactive=False, height=680)
                upload_btn = gr.File(label='Upload a PDF', file_types=[".pdf"])
    return demo

if __name__ == "__main__":
    demo = createUI()
    demo.launch(pwa=True, share=False)