# import streamlit as st
# import tempfile
# import os
# from gtts import gTTS
# from deep_translator import GoogleTranslator
# from agents.finance_chatbot_agent import multilingual_chatbot, voice_input_to_text

# st.header("💬 Finance Chatbot")

# # Language dropdown for Finance Chatbot
# lang_map = {
#     "English": "en",
#     "Hindi": "hi",
#     "Gujarati": "gu"
# }
# selected_lang = st.selectbox("Select Language for Chatbot Response:", list(lang_map.keys()))

# # Helper function to get answer in selected language with translation
# def get_answer_in_selected_language(question, lang_code):
#     if lang_code == "en":
#         # For English, directly call your multilingual_chatbot (which already supports English)
#         return multilingual_chatbot(question)
#     else:
#         # Translate question to English
#         question_in_en = GoogleTranslator(source=lang_code, target="en").translate(question)
#         # Get English answer from chatbot
#         answer_in_en = multilingual_chatbot(question_in_en)
#         # Translate answer back to selected language
#         answer_in_lang = GoogleTranslator(source="en", target=lang_code).translate(answer_in_en)
#         return answer_in_lang

# # User text input
# user_question = st.text_area("Enter your finance-related question:", height=150)

# # Text-based Q&A
# if st.button("Get Answer (Multilingual)"):
#     if user_question.strip():
#         answer = get_answer_in_selected_language(user_question, lang_map[selected_lang])
#         st.markdown("**Answer:**")
#         st.write(answer)

#         # Text to Speech output
#         tts = gTTS(text=answer, lang=lang_map[selected_lang])
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
#             tts.save(tmp_audio.name)
#             audio_bytes = open(tmp_audio.name, "rb").read()
#         st.audio(audio_bytes, format="audio/mp3")
#         os.remove(tmp_audio.name)

#     else:
#         st.warning("Please enter a question.")

# # Voice-based Q&A
# if st.button("🎙️ Use Voice Input"):
#     voice_text = voice_input_to_text()
#     st.write(f"🎤 You said: {voice_text}")
#     if voice_text and not voice_text.startswith("Sorry") and not voice_text.startswith("Speech recognition failed"):
#         answer = get_answer_in_selected_language(voice_text, lang_map[selected_lang])
#         st.markdown("**Answer:**")
#         st.write(answer)

#         # Text to Speech output
#         tts = gTTS(text=answer, lang=lang_map[selected_lang])
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
#             tts.save(tmp_audio.name)
#             audio_bytes = open(tmp_audio.name, "rb").read()
#         st.audio(audio_bytes, format="audio/mp3")
#         os.remove(tmp_audio.name)
#     else:
#         st.warning(voice_text)































# import streamlit as st
# import tempfile
# import os
# from gtts import gTTS
# from deep_translator import GoogleTranslator
# from agents.finance_chatbot_agent import multilingual_chatbot, voice_input_to_text

# # --- Page Config ---
# st.set_page_config(page_title="Finance Chatbot", layout="centered")

# # --- Custom CSS Styling ---
# st.markdown("""
#     <style>
#         /* Clean background */
#         .stApp {
#             background-color: white;
#         }

#         /* Title Styling */
#         .title {
#             font-size: 40px;
#             font-weight: bold;
#             color: #2E86C1; /* Blue accent */
#             text-align: center;
#             margin-bottom: 5px;
#         }

#         .subtitle {
#             text-align: center;
#             font-size: 18px;
#             color: #7D3C98; /* Purple accent */
#             margin-bottom: 25px;
#         }

#         /* Input box styling */
#         textarea {
#             border: 2px solid #2E86C1 !important;
#             border-radius: 10px !important;
#             font-size: 16px !important;
#             padding: 10px !important;
#         }

#         /* Buttons */
#         div.stButton > button {
#             background: linear-gradient(90deg, #3498DB, #2E86C1);
#             color: white;
#             font-size: 16px;
#             font-weight: bold;
#             border-radius: 10px;
#             padding: 8px 16px;
#             border: none;
#             margin-top: 10px;
#         }
#         div.stButton > button:hover {
#             background: linear-gradient(90deg, #5DADE2, #2874A6);
#             color: white;
#         }

#         /* Dropdown label */
#         label {
#             font-weight: bold !important;
#             color: #2C3E50 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # --- Title ---
# st.markdown("<div class='title'>💬 Finance Chatbot</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Your AI-powered Finance Assistant</div>", unsafe_allow_html=True)

# # --- Language dropdown ---
# lang_map = {
#     "English": "en",
#     "Hindi": "hi",
#     "Gujarati": "gu"
# }
# selected_lang = st.selectbox("🌐 Select Language for Chatbot Response:", list(lang_map.keys()))

# # --- Helper function ---
# def get_answer_in_selected_language(question, lang_code):
#     if lang_code == "en":
#         return multilingual_chatbot(question)
#     else:
#         question_in_en = GoogleTranslator(source=lang_code, target="en").translate(question)
#         answer_in_en = multilingual_chatbot(question_in_en)
#         answer_in_lang = GoogleTranslator(source="en", target=lang_code).translate(answer_in_en)
#         return answer_in_lang

# # --- User text input ---
# user_question = st.text_area("📝 Enter your finance-related question:", height=150)

# # --- Text-based Q&A ---
# if st.button("💡 Get Answer (Multilingual)"):
#     if user_question.strip():
#         answer = get_answer_in_selected_language(user_question, lang_map[selected_lang])
#         st.markdown("**Answer:**")
#         st.write(answer)

#         # Text to Speech output
#         tts = gTTS(text=answer, lang=lang_map[selected_lang])
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
#             tts.save(tmp_audio.name)
#             audio_bytes = open(tmp_audio.name, "rb").read()
#         st.audio(audio_bytes, format="audio/mp3")
#         os.remove(tmp_audio.name)
#     else:
#         st.warning("⚠️ Please enter a question.")

# # --- Voice-based Q&A ---
# if st.button("🎙️ Use Voice Input"):
#     voice_text = voice_input_to_text()
#     st.write(f"🎤 You said: {voice_text}")
#     if voice_text and not voice_text.startswith("Sorry") and not voice_text.startswith("Speech recognition failed"):
#         answer = get_answer_in_selected_language(voice_text, lang_map[selected_lang])
#         st.markdown("**Answer:**")
#         st.write(answer)

#         tts = gTTS(text=answer, lang=lang_map[selected_lang])
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
#             tts.save(tmp_audio.name)
#             audio_bytes = open(tmp_audio.name, "rb").read()
#         st.audio(audio_bytes, format="audio/mp3")
#         os.remove(tmp_audio.name)
#     else:
#         st.warning(voice_text)























# import streamlit as st
# import tempfile
# import os
# from gtts import gTTS
# from deep_translator import GoogleTranslator
# from agents.finance_chatbot_agent import multilingual_chatbot, voice_input_to_text

# # --- Page Config ---
# st.set_page_config(page_title="Finance Chatbot", layout="centered")

# # --- Custom CSS Styling ---
# st.markdown("""
#     <style>
#         /* Clean background */
#         .stApp {
#             background-color: white;
#         }

#         /* Title Styling */
#         .title {
#             font-size: 40px;
#             font-weight: bold;
#             color: #2E86C1; /* Blue accent */
#             text-align: center;
#             margin-bottom: 5px;
#         }

#         .subtitle {
#             text-align: center;
#             font-size: 18px;
#             color: #7D3C98; /* Purple accent */
#             margin-bottom: 25px;
#         }

#         /* Input box styling */
#         textarea {
#             border: 2px solid #2E86C1 !important;
#             border-radius: 10px !important;
#             font-size: 16px !important;
#             padding: 10px !important;
#         }

#         /* Buttons */
#         div.stButton > button {
#             background: linear-gradient(90deg, #3498DB, #2E86C1);
#             color: white;
#             font-size: 16px;
#             font-weight: bold;
#             border-radius: 10px;
#             padding: 8px 16px;
#             border: none;
#             margin-top: 10px;
#         }
#         div.stButton > button:hover {
#             background: linear-gradient(90deg, #5DADE2, #2874A6);
#             color: white;
#         }

#         /* Dropdown label */
#         label {
#             font-weight: bold !important;
#             color: #2C3E50 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # --- Title ---
# st.markdown("<div class='title'>💬 Finance Chatbot</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Your AI-powered Finance Assistant</div>", unsafe_allow_html=True)

# # --- Language dropdown ---
# lang_map = {
#     "English": "en",
#     "Hindi": "hi",
#     "Gujarati": "gu"
# }
# selected_lang = st.selectbox("🌐 Select Language for Chatbot Response:", list(lang_map.keys()))

# # --- Helper function ---
# def get_answer_in_selected_language(question, lang_code):
#     if lang_code == "en":
#         return multilingual_chatbot(question)
#     else:
#         # Translate question to English
#         question_in_en = GoogleTranslator(source=lang_code, target="en").translate(question)
#         # Get answer from model
#         answer_in_en = multilingual_chatbot(question_in_en)
#         # Translate answer back to selected language
#         answer_in_lang = GoogleTranslator(source="en", target=lang_code).translate(answer_in_en)
#         return answer_in_lang

# # --- User text input ---
# user_question = st.text_area("📝 Enter your finance-related question:", height=150)

# # --- Text-based Q&A ---
# if st.button("💡 Get Answer (Multilingual)"):
#     if user_question.strip():
#         answer = get_answer_in_selected_language(user_question, lang_map[selected_lang])
#         st.markdown("**Answer:**")
#         st.write(answer)

#         # Text to Speech output
#         tts = gTTS(text=answer, lang=lang_map[selected_lang])
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
#             tts.save(tmp_audio.name)
#             audio_bytes = open(tmp_audio.name, "rb").read()
#         st.audio(audio_bytes, format="audio/mp3")
#         os.remove(tmp_audio.name)
#     else:
#         st.warning("⚠️ Please enter a question.")

# # --- Voice-based Q&A ---
# if st.button("🎙️ Use Voice Input"):
#     voice_text = voice_input_to_text()
#     st.write(f"🎤 You said: {voice_text}")
#     if voice_text and not voice_text.startswith("Sorry") and not voice_text.startswith("Speech recognition failed"):
#         answer = get_answer_in_selected_language(voice_text, lang_map[selected_lang])
#         st.markdown("**Answer:**")
#         st.write(answer)

#         tts = gTTS(text=answer, lang=lang_map[selected_lang])
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
#             tts.save(tmp_audio.name)
#             audio_bytes = open(tmp_audio.name, "rb").read()
#         st.audio(audio_bytes, format="audio/mp3")
#         os.remove(tmp_audio.name)
#     else:
#         st.warning(voice_text)








import streamlit as st
import tempfile
import os
from gtts import gTTS
from deep_translator import GoogleTranslator
from agents.finance_chatbot_agent import multilingual_chatbot, voice_input_to_text

# --- Page Config ---
st.set_page_config(page_title="Finance Chatbot", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: white;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7D3C98;
            margin-bottom: 25px;
        }
        textarea {
            border: 2px solid #2E86C1 !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            padding: 10px !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #3498DB, #2E86C1);
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
            margin-top: 10px;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #5DADE2, #2874A6);
            color: white;
        }
        label {
            font-weight: bold !important;
            color: #2C3E50 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<div class='title'>💬 Finance Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'> AI-powered Finance Assistant</div>", unsafe_allow_html=True)

# --- Language dropdown ---
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu"
}
selected_lang = st.selectbox("🌐 Select Language for Chatbot Response:", list(lang_map.keys()))
lang_code = lang_map[selected_lang]

# --- Translation Helper ---
def translate_text(text, target_language):
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        st.warning(f"⚠️ Translation failed: {e}")
        return text

# --- Main function ---
def get_answer_in_selected_language(question, lang_code):
    if lang_code == "en":
        return multilingual_chatbot(question)
    else:
        question_in_en = translate_text(question, "en")
        answer_in_en = multilingual_chatbot(question_in_en)
        answer_in_target = translate_text(answer_in_en, lang_code)
        return answer_in_target

# --- User text input ---
user_question = st.text_area("📝 Enter your finance-related question:", height=150)

# --- Text-based Q&A ---
if st.button("💡 Get Answer (Multilingual)"):
    if user_question.strip():
        answer = get_answer_in_selected_language(user_question, lang_code)
        st.markdown("**Answer:**")
        st.write(answer)

        # Text to Speech output
        try:
            tts = gTTS(text=answer, lang=lang_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tts.save(tmp_audio.name)
                audio_bytes = open(tmp_audio.name, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
            os.remove(tmp_audio.name)
        except Exception as e:
            st.warning("⚠️ Text-to-speech failed.")
    else:
        st.warning("⚠️ Please enter a question.")

# --- Voice-based Q&A ---
if st.button("🎙️ Use Voice Input"):
    voice_text = voice_input_to_text()
    st.write(f"🎤 You said: {voice_text}")
    if voice_text and not voice_text.startswith("Sorry") and not voice_text.startswith("Speech recognition failed"):
        answer = get_answer_in_selected_language(voice_text, lang_code)
        st.markdown("**Answer:**")
        st.write(answer)

        try:
            tts = gTTS(text=answer, lang=lang_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tts.save(tmp_audio.name)
                audio_bytes = open(tmp_audio.name, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
            os.remove(tmp_audio.name)
        except Exception as e:
            st.warning("⚠️ Text-to-speech failed.")
    else:
        st.warning(voice_text)
