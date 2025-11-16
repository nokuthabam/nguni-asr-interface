# Nguni ASR Demo
Interface to Test Nguni ASR Models finetuned using whisper and wav2vec2

A multilingual Nguni ASR interface supporting:
- isiZulu (zu_wav2vec2)
- isiXhosa (zu_wav2vec2)
- siSwati (ssw_wav2vec2)
- Multilingual Nguni model (nguni_wav2vec2)

This space loads models directly from Hugging Face Hub using 'pipeline()'


## Features
- Microphone recording
- Audio upload
- Dynamic model selection
- Runs on CPU inside Hugging Face Spaces

## How to use
1. Select a language/model
2. Record or upload audio
3. Receive transcription
