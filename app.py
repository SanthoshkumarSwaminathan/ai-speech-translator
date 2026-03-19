import streamlit as st
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import os

st.set_page_config(page_title="AI Translator", layout="centered")

st.title("🎤 AI Speech Translator")
st.markdown("Convert English speech → Tamil text + audio")

# Load models (cached)
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return whisper_model, tokenizer, translator_model

whisper_model, tokenizer, translator_model = load_models()

uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])

if uploaded_file is not None:

    with st.spinner("🔍 Processing... Please wait"):
        # Save temp
        temp_audio = "temp.wav"
        with open(temp_audio, "wb") as f:
            f.write(uploaded_file.read())

        # Transcribe
        result = whisper_model.transcribe(temp_audio)
        text = result["text"].strip()

        # Translate
        inputs = tokenizer(text, return_tensors="pt")

        translated_tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml"),
            max_length=400
        )

        translated_text = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]

        # TTS
        tts = gTTS(text=translated_text, lang='ta')
        output_audio = "output.mp3"
        tts.save(output_audio)

    # Display nicely
    st.success("✅ Processing Complete")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Original Text")
        st.write(text)

    with col2:
        st.subheader("🌍 Tamil Translation")
        st.write(translated_text)

    st.subheader("🔊 Audio Output")
    st.audio(output_audio)