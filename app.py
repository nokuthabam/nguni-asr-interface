import gradio as gr
import librosa
import torch
import os
from pydub import AudioSegment
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder


# Model Cache
whisper_cache = {}
wav2vec2_cache = {}


# audio conversion
def convert_audio(input_file, target_sr=16000):
    # Converts mp3 or other formats to wav with target sample rate
    if input_file.endswith(".wav"):
        return input_file
    
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    output_file = "converted_audio.wav"
    audio.export(output_file, format="wav")
    return output_file


# Model Definitions
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
        "repo": "nmoyo45/xh_wav2vec2",
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


def wav2vec2_ctc_beam(repo, audio_path, beams):
    """
    Transcribe audio using Wav2Vec2 with CTC beam search decoding.
    """

    if repo not in wav2vec2_cache:
        processor = Wav2Vec2Processor.from_pretrained(repo)
        model = Wav2Vec2ForCTC.from_pretrained(repo)
        vocab = list(processor.tokenizer.get_vocab().keys())
        decoder = build_ctcdecoder(vocab)
        wav2vec2_cache[repo] = (processor, model, decoder)

    processor, model, decoder = wav2vec2_cache[repo]

    speech, sample_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=sample_rate, padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    logits = logits.squeeze(0).cpu().numpy()
    transcription = decoder.decode(logits, beam_width=beams)
    return transcription


def transcribe_audio(language, audio_path, beams):
    """
    Transcribe audio using the specified model.
    """
    if audio_path is None:
        return "Please upload or record an audio file."
    
    wav_path = convert_audio(audio_path)

    repo = models[language]["repo"]

    if "whisper" in language.lower():

        if repo not in whisper_cache:
            whisper_cache[repo] = pipeline("automatic-speech-recognition", model=repo)

        asr = whisper_cache[repo]
        return asr(
            wav_path,
            generate_kwargs={"num_beams": beams}
        )["text"]
    else:
        return wav2vec2_ctc_beam(repo, wav_path, beams)


with gr.Blocks() as demo:
    gr.Markdown("# Multilingual Speech-to-Text Transcription")
    gr.Markdown(
        "Transcribe audio in isiZulu, isiXhosa, siSwati, and isiNdebele using Wav2Vec2 (CTC Beam Search) and Whisper (Seq2Seq Beam Search)."
    )

    beams = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Beams")

    language = gr.Dropdown(choices=list(models.keys()), label="Select Language")

    prompt_display = gr.Markdown()

    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record or Upload Audio")
    output = gr.Textbox(label="Transcription")
    language.change(fn=show_prompts, inputs=language, outputs=prompt_display)
    audio.change(fn=transcribe_audio, inputs=[language, audio, beams], outputs=output)


if __name__ == "__main__":
    demo.launch()