import gradio as gr
from transformers import pipeline

models = {
    "isiZulu": "nmoyo45/zu_wav2vec2",
    "isiXhosa": "nmoyo45/xh_wav2vec2",
    "siSwati": "nmoyo45/sv_wav2vec2",
    "isiNdebele": "nmoyo45/nbl_wav2vec2",
}
KENLM_PATH = "nguni_3gram.arpa"

def transcribe_audio(language, audio):
    """
    Transcribe audio using the specified model.
    """

    model_id = models[language]
    asr = pipeline("automatic-speech-recognition", model=model_id)
    transcription = asr(audio)["text"]
    return transcription

demo = gr.Interface(
    fn =transcribe_audio,
    inputs=[
        gr.Dropdown(choices=list(models.keys()), label="Select Language"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record or Upload Audio")
    ],
    outputs = "textbox",
    title="Multilingual Speech-to-Text Transcription",
    description="Transcribe audio in isiZulu, isiXhosa, siSwati, and isiNdebele using pre-trained models."
)

if __name__ == "__main__":
    demo.launch()