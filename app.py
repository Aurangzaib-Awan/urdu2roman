import gradio as gr
from inference import translate_sentence
from tokenizer import urdu2idx, roman2idx, idx2roman
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(urdu_text):
    output = translate_sentence(None, urdu_text, urdu2idx, idx2roman)  # model is loaded inside inference.py
    return output

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Type Urdu sentence here..."),
    outputs="text",
    title="Urdu to Roman Converter",
    description="Type an Urdu sentence and get the Romanized version."
)

iface.launch()
