import gradio as gr
from vision import Vision2Text
from audio import Text2Audio

vision = Vision2Text()
audio = Text2Audio()

def generate_compliment(image):
    image_description = vision.generate_description(image)
    compliment_text = vision.generate_compliment(image, image_description)
    compliment_audio = audio.generate_audio_from_text(compliment_text)
    return image_description, compliment_text, compliment_audio

demo = gr.Interface(
    fn=generate_compliment,
    inputs=[gr.Image(label="Upload an image", type="pil")],
    outputs=[gr.Textbox(label="Image Description"), gr.Textbox(label="Generated Compliment"), gr.Audio(label="Generated Audio")],
    title="Image to Compliment Generator",
    description="Upload an image to generate a compliment",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)
