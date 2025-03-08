import gradio as gr
import os

import fitz
from PIL import Image

with gr.Blocks() as demo:
    with gr.Row():
        api_key = gr.Textbox(
            placeholder='Enter OpenAI API key',
            interactive=True,
            label='API Key'
        )
        change_api_key = gr.Button('Change Key')

    with gr.Row():
        chatbot = gr.Chatbot(label='Chatbot', type='messages', height=650)
        pdf_display = gr.Image(label='Uploaded PDF Page', interactive=False, height=680)
    
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter"
        )
        submit_btn = gr.Button('Submit')
        upload_btn = gr.File(label='Upload a PDF', file_types=[".pdf"])

if __name__ == "__main__":
    demo.launch()