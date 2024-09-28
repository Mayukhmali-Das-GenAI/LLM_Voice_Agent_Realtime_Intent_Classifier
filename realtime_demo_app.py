import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sounddevice as sd
import numpy as np
import whisper
import time
import os
import pygame

pygame.mixer.init()

@st.cache_resource
def load_intent_model():
    model = BertForSequenceClassification.from_pretrained('intent_model')
    tokenizer = BertTokenizer.from_pretrained('intent_model')
    return model, tokenizer

@st.cache_resource
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

def record_audio(duration, samplerate=16000):
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    
    progress_bar = st.progress(0)
    
    for i in range(duration):
        time.sleep(1)
        progress_bar.progress((i + 1) / duration)
    
    sd.wait()
    progress_bar.empty()
    return recording

def transcribe_audio(audio, model):
    audio = whisper.pad_or_trim(audio.flatten())
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

def predict_intent(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item()
    intent = "LLM" if prediction == 0 else "Non-LLM"
    return intent, confidence

def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def main():
    st.set_page_config(page_title="Voice Agent Intent Dashboard", layout="wide")
    
    st.title("Voice Agent Intent Dashboard")
    
    intent_model, tokenizer = load_intent_model()
    
    st.sidebar.header("Settings")
    num_conversations = st.sidebar.number_input("Number of Conversations", min_value=1, max_value=10, value=4)
    whisper_model_name = st.sidebar.selectbox("Whisper Model", ["base", "medium"])
    recording_duration = st.sidebar.slider("Recording Duration (seconds)", min_value=4, max_value=30, value=10)
    
    whisper_model = load_whisper_model(whisper_model_name)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversation Progress")
        progress_bar = st.progress(0)
        
        if st.button("Start Conversations"):
            for i in range(st.session_state.current_conversation, num_conversations):
                st.session_state.current_conversation = i + 1
                progress_bar.progress((i + 1) / num_conversations)
                
                st.write(f"Recording Conversation {i+1}...")
                audio = record_audio(recording_duration)
                
                transcription = transcribe_audio(audio, whisper_model)
                
                context = " ".join([convo['transcription'] for convo in st.session_state.conversation_history])
                full_text = context + " " + transcription if context else transcription
                
                st.write("Text input to intent model:")
                st.text(full_text)
                
                if len(transcription.split()) < 2:
                    intent = "Non-LLM"
                    confidence = 1.0
                    st.write("Transcription too short. Playing background audio.")
                else:
                    intent, confidence = predict_intent(full_text, intent_model, tokenizer)
                
                st.session_state.conversation_history.append({
                    'transcription': transcription,
                    'intent': intent,
                    'confidence': confidence
                })
                
                st.write("---")
                st.write(f"Conversation {i+1} Details:")
                st.write(f"Transcript: {transcription}")
                st.write(f"Predicted Intent: {intent}")
                st.write(f"Confidence Score: {confidence:.4f}")
                st.write("---")
                
                if intent == "LLM" and len(transcription.split()) >= 2:
                    play_audio("audio_files/answer.mp3")
                else:
                    play_audio("audio_files/bg.mp3")
                
                time.sleep(1)
            
            st.success("All conversations completed!")
    
    with col2:
        st.subheader("Conversation History")
        for i, convo in enumerate(st.session_state.conversation_history, 1):
            with st.expander(f"Conversation {i}"):
                st.write(f"Transcription: {convo['transcription']}")
                st.write(f"Predicted Intent: {convo['intent']}")
                st.write(f"Confidence Score: {convo['confidence']:.4f}")
                st.write("---")
    
    if st.button("Reset Session"):
        reset_session()
        st.experimental_rerun()

if __name__ == "__main__":
    main()