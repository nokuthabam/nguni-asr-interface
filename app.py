import gradio as gr
from transformers import pipeline

models = {
    "isiZulu Wav2Vec2": {
        "repo":"nmoyo45/zu_wav2vec2",
        "prompts": [
            "Ngikhuluma isiZulu",
            "Sawubona, unjani?",
            "Umakoti omuhle",
            "Wena Wendlovu",
            "Ngicela ungikhombise indlela",
        ]},
    "isiXhosa Wav2Vec2": {
        "repo":"nmoyo45/xh_wav2vec2",
        "prompts": [
            "Ndifuna ukuthetha isiXhosa",
            "Molo, unjani?",
            "Makungabikho mntu okuvisa ubuhlungu",
            "Uxolo andivanga",
            "As'phelelanga"
        ]},
    "siSwati Wav2Vec2": {
        "repo": "nmoyo45/sv_wav2vec2",
        "prompts": [
            "Ngicela emanti",
            "Uphi lomngani wami?",
            "Sicela lusito lwakho",
            "Lo moya uvunguta ngemandla",
            "Ngitokubona kusasa",
        ]},
    "isiNdebele Wav2Vec2": {
        "repo": "nmoyo45/nbl_wav2vec2", 
        "prompts": [
            "Lotjhani",
            "Siyalemukela",
            "Ngikhona ngiyathokoza",
            "Ibizo lakho ngubani",
            "Ukutjho njani lokhu ngesiNdebele",
        ]},
    "Multilingual Wav2Vec2": {
        "repo": "nmoyo45/multilingual_wav2vec2", 
        "prompts": [
            "Ngicela ungikhombise indlela",
            "Uxolo andivanga",
            "Sicela lusito lwakho",
            "Ngikhona ngiyathokoza",
        ]},
    "isiZulu Whisper": {
        "repo": "nmoyo45/zu_whisper", 
        "prompts": [
            "Ngikhuluma isiZulu",
            "Sawubona, unjani?",
            "Umakoti omuhle",
            "Wena Wendlovu",
            "Ngicela ungikhombise indlela",
        ]},
    "isiXhosa Whisper": {
        "repo": "nmoyo45/xh_whisper", 
        "prompts": [
            "Ndifuna ukuthetha isiXhosa",
            "Molo, unjani?",
            "Makungabikho mntu okuvisa ubuhlungu",
            "Sihleli thina ngokungazi",
            "As'phelelanga"
        ]},
    "siSwati Whisper": {
        "repo": "nmoyo45/ss_whisper", 
        "prompts": [
            "Ngicela emanti",
            "Uphi lomngani wami?",
            "Sicela lusito lwakho",
            "Lo moya uvunguta ngemandla",
            "Ngitokubona kusasa",
        ]},
    "isiNdebele Whisper": {
        "repo": "nmoyo45/nr_whisper", 
        "prompts": [
            "Lotjhani",
            "Siyalemukela",
            "Ngikhona ngiyathokoza",
            "Ibizo lakho ngubani",
            "Ukutjho njani lokhu ... ngesiNdebele",
        ]},
    "Multilingual Whisper": {
        "repo": "nmoyo45/multilingual_whisper", 
        "prompts": [
            "Ngicela ungikhombise indlela",
            "Uxolo andivanga",
            "Sicela lusito lwakho",
            "Ngikhona ngiyathokoza",
        ]}
}


def show_prompts(language):
    """
    Display example prompts for the selected language model.
    """
    items = models[language]["prompts"]
    formatted_prompts = "\n".join(f"- {item}" for item in items)
    return formatted_prompts


def transcribe_audio(language, audio):
    """
    Transcribe audio using the specified model.
    """

    model_id = models[language]["repo"]
    asr = pipeline("automatic-speech-recognition", model=model_id)
    transcription = asr(audio)["text"]
    return transcription

with gr.Blocks() as demo:
    gr.Markdown("# Multilingual Speech-to-Text Transcription")
    gr.Markdown("Transcribe audio in isiZulu, isiXhosa, siSwati, and isiNdebele using pre-trained models.")
    language = gr.Dropdown(choices=list(models.keys()), label="Select Language")
    prompt_display = gr.Markdown()
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record or Upload Audio")
    output = gr.Textbox(label="Transcription")
    language.change(fn=show_prompts, inputs=language, outputs=prompt_display)
    audio.change(fn=transcribe_audio, inputs=[language, audio], outputs=output)

if __name__ == "__main__":
    demo.launch()