import gradio as gr
import librosa
import torch
import torchaudio
import os
from pydub import AudioSegment
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
import uuid


# Model Cache
whisper_cache = {}
wav2vec2_cache = {}
# KENLM_PATH = "nguni_3gram.arpa"

# audio conversion
def convert_audio(input_file, target_sr=16000):
    if input_file.endswith(".wav"):
        return input_file
    
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    output_file = f"converted_{uuid.uuid4().hex}.wav"
    audio.export(output_file, format="wav")
    return output_file


# Model Definitions
models = {
    "isiZulu Wav2Vec2": {
        "repo":"nmoyo45/zu_wav2vec2",
        "LM_PATH": "zulu_3gram.arpa",
        "prompts": [
            "Ngikhuluma isiZulu",
            "Sawubona, unjani?",
            "Umakoti omuhle",
            "Wena Wendlovu",
            "Ngicela ungikhombise indlela",
        ]},
    "isiXhosa Wav2Vec2": {
        "repo": "nmoyo45/xh_wav2vec2",
        "LM_PATH": "xhosa_3gram.arpa",
        "prompts": [
            "Ndifuna ukuthetha isiXhosa",
            "Molo, unjani?",
            "Makungabikho mntu okuvisa ubuhlungu",
            "Uxolo andivanga",
            "As'phelelanga"
        ]},
    "siSwati Wav2Vec2": {
        "repo": "nmoyo45/sv_wav2vec2",
        "LM_PATH": "siswati_3gram.arpa",
        "prompts": [
            "Ngicela emanti",
            "Uphi lomngani wami?",
            "Sicela lusito lwakho",
            "Lo moya uvunguta ngemandla",
            "Ngitokubona kusasa",
        ]},
    "isiNdebele Wav2Vec2": {
        "repo": "nmoyo45/nbl_wav2vec2", 
        "LM_PATH": "ndebele_3gram.arpa",
        "prompts": [
            "Lotjhani",
            "Siyalemukela",
            "Ngikhona ngiyathokoza",
            "Ibizo lakho ngubani",
            "Ukutjho njani lokhu ngesiNdebele",
        ]},
    "Multilingual Wav2Vec2": {
        "repo": "nmoyo45/multilingual_wav2vec2", 
        "LM_PATH": "nguni_3gram.arpa",
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


def wav2vec2_ctc_beam(repo, audio_path, beams, language):
    """
    Correct Wav2Vec2 beam-search decoding using pyctcdecode + KenLM.
    """

    if repo not in wav2vec2_cache:
        processor = Wav2Vec2Processor.from_pretrained(repo)
        model = Wav2Vec2ForCTC.from_pretrained(repo)

        vocab_dict = processor.tokenizer.get_vocab()
        vocab = sorted(vocab_dict.keys(), key=lambda x: vocab_dict[x])
        if "LM_PATH" in models[language]:
            kenlm_path = models[language]["LM_PATH"]
            if os.path.exists(kenlm_path):
                decoder = build_ctcdecoder(
                    labels=vocab, 
                    kenlm_model_path=kenlm_path,
                    alpha=0.5,
                    beta=1.5)
            else:
                print(f"KenLM model not found at {kenlm_path}. Proceeding without language model.")
                decoder = build_ctcdecoder(labels=vocab)
        else:
            decoder = build_ctcdecoder(labels=vocab)
        wav2vec2_cache[repo] = (processor, model, decoder)
    processor, model, decoder = wav2vec2_cache[repo]

    # Load with torchaudio (safer than librosa)
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # Shape [1, T] → [T]
    waveform = waveform.squeeze()

    # Prepare input
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Run model
    with torch.no_grad():
        logits = model(inputs["input_values"]).logits.cpu().numpy()[0]

    # Beam search decode
    transcription = decoder.decode(logits, beam_width=beams)
    return transcription.lower()


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
        return wav2vec2_ctc_beam(repo, wav_path, beams, language)


with gr.Blocks() as demo:
    gr.Markdown("# Multilingual waveform-to-Text Transcription")
    gr.Markdown(
        "Transcribe audio in isiZulu, isiXhosa, siSwati, and isiNdebele using Wav2Vec2 (CTC Beam Search) and Whisper (Seq2Seq Beam Search)."
    )

    # Set number of beams to 5
    beams = gr.Slider(1, 10, value=5, step=1, label="Number of Beams for Decoding")
    language = gr.Dropdown(choices=list(models.keys()), label="Select Language")

    prompt_display = gr.Markdown()

    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record or Upload Audio")
    output = gr.Textbox(label="Transcription")
    language.change(fn=show_prompts, inputs=language, outputs=prompt_display)
    audio.change(fn=transcribe_audio, inputs=[language, audio, beams], outputs=output)


if __name__ == "__main__":
    demo.launch()