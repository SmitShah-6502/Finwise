# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from config import GROQ_API_KEY

# from deep_translator import GoogleTranslator
# from langdetect import detect
# import speech_recognition as sr

# MODEL_NAME = "llama-3.1-8b-instant"

# llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# # Prompt for financial assistant
# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""
# You are a knowledgeable financial assistant. Provide clear and concise answers to the user's finance-related questions.

# User question: {question}
# Answer:
# """
# )

# chain = LLMChain(llm=llm, prompt=prompt)

# # ✅ Core function
# def finance_chatbot_answer(question):
#     return chain.run({"question": question})

# # ✅ Multilingual support (detect + translate)
# def multilingual_chatbot(question):
#     detected_lang = detect(question)

#     if detected_lang != 'en':
#         translated = GoogleTranslator(source='auto', target='en').translate(question)
#         english_answer = finance_chatbot_answer(translated)
#         back_translated = GoogleTranslator(source='en', target=detected_lang).translate(english_answer)
#         return f"[{detected_lang.upper()}] {back_translated}"
#     else:
#         return finance_chatbot_answer(question)

# # ✅ Voice input function
# def voice_input_to_text():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("🎙️ Listening... Speak now.")
#         audio = recognizer.listen(source)

#         try:
#             text = recognizer.recognize_google(audio)
#             print(f"📝 Recognized: {text}")
#             return text
#         except sr.UnknownValueError:
#             return "Sorry, I couldn't understand your voice."
#         except sr.RequestError as e:
#             return f"Speech recognition failed: {e}"















# # agents/finance_chatbot_agent.py
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel
# import torch
# import speech_recognition as sr

# # ---------------------------
# # 1. Load Base Model & Tokenizer
# # ---------------------------
# base_model_name = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# # ---------------------------
# # 2. Load LoRA-finetuned Weights
# # ---------------------------
# model = PeftModel.from_pretrained(base_model, "./finetuned-flan-t5-lora")
# model.eval()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # ---------------------------
# # 3. Chatbot Function
# # ---------------------------
# def multilingual_chatbot(question: str) -> str:
#     """
#     Generates a finance answer for a given question using the LoRA-finetuned model.
#     Produces longer, paragraph-style, informative responses.
#     """
#     inputs = tokenizer(
#         question,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512  # allows longer questions
#     ).to(device)
    
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_length=256,          # output up to 256 tokens (paragraph)
#             num_beams=5,             # beam search for better quality
#             no_repeat_ngram_size=3,  # avoid repeated phrases
#             early_stopping=True,
#             do_sample=True,          # allow some creativity
#             top_p=0.95,              # nucleus sampling
#             temperature=0.7          # controls randomness
#         )
    
#     answer = tokenizer.decode(output[0], skip_special_tokens=True)
#     return answer



# # ---------------------------
# # 4. Voice Input Function
# # ---------------------------
# def voice_input_to_text() -> str:
#     """
#     Records audio from microphone and converts it to text using SpeechRecognition.
#     Returns recognized text or an error message if failed.
#     """
#     r = sr.Recognizer()
#     try:
#         with sr.Microphone() as source:
#             print("🎤 Listening...")
#             r.adjust_for_ambient_noise(source)
#             audio = r.listen(source)
#         text = r.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         return "Sorry, I could not understand the audio."
#     except sr.RequestError:
#         return "Speech recognition failed. Check your internet connection."














# agents/finance_chatbot_agent.py
import os

try:
    import streamlit as st
    _secrets = getattr(st, "secrets", {})
    GROQ_API_KEY = _secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Optional: minimal translation without adding heavy deps.
# If you have deep_translator already in requirements, you can uncomment.
# from deep_translator import GoogleTranslator

try:
    from groq import Groq
except Exception:
    Groq = None

MODEL_NAME = "llama-3.3-70b-versatile"

def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY or Groq is None:
        return (
            "LLM is disabled (no GROQ_API_KEY set on the server). "
            "Add it in Streamlit → Settings → Secrets to enable answers."
        )
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def multilingual_chatbot(user_text: str, target_lang: str = "en") -> str:
    """
    Returns an answer in English by default.
    If you want translation, pass target_lang (e.g., 'hi', 'gu') and add deep_translator to requirements.
    """
    answer_en = _call_groq(user_text)

    # If you need multilingual support, uncomment and add deep-translator in requirements.txt
    # if target_lang and target_lang != "en":
    #     try:
    #         return GoogleTranslator(source="en", target=target_lang).translate(answer_en)
    #     except Exception:
    #         pass

    return answer_en

def voice_input_to_text() -> str:
    """
    Voice input disabled on Streamlit Cloud Lite build.
    Keep function for compatibility.
    """
    return "Voice input is disabled on this deployment."
