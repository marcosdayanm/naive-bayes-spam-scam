import random
from typing import Tuple

from sklearn.naive_bayes import MultinomialNB
import streamlit as st

from nb import load_model_and_vectorizer, predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# MODEL_PATH = Path("artifacts/model.joblib")

model, vectorizer = None, None 



def classify(model: MultinomialNB, vectorizer: TfidfVectorizer, text: str) -> Tuple[str, str]:
    pred = predict(model, vectorizer, text)
    tag = "scam" if pred == 1 else "spam"
    color = "#e53935" if tag == "scam" else "#fbc02d"  # rojo para spam, amarillo para scam
    return tag, color


def app(model: MultinomialNB, vectorizer: TfidfVectorizer):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tags" not in st.session_state:
        st.session_state.tags = []

    st.set_page_config(
        page_title="Spam or Scam (Naive Bayes)",
        page_icon="📬",
        layout="centered",
    )

    st.header("Clasificador Spam o Scam — Naive Bayes")
    st.subheader("¿El mensaje que recibiste es spam o scam?")
    st.caption("Marcos Dayan Mann - A01782876")
    st.caption("Paúl Araque Fernández - A01027626")

    with st.form("form_clasificacion"):
        message = st.text_area(
            "Pega el mensaje a evaluar",
            height=160,
            placeholder='Ej.: "Has ganado un premio, da clic aquí..."',
            key="message_input"
        )
        submitted = st.form_submit_button("Clasificar")

    if submitted:
        if not message.strip():
            st.warning("Por favor, ingresa un mensaje antes de clasificar.")
        else:
            tag, color = classify(model, vectorizer, message)
            st.subheader("Resultado")
            st.markdown(
                f"""
                <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:stretch;">
                    <div style="flex:1; min-width:240px; text-align:center; padding:0.75em; border:1px solid #ddd; border-radius:8px;">
                        <div style="font-size:16px; color:gray; margin-bottom:4px;">Clasificación</div>
                        <div style="font-size:32px; font-weight:bold; color:{color}; line-height:1.1;">
                            {tag.upper()}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.session_state.messages.append(message)
            st.session_state.tags.append(tag)

    st.divider()
    st.subheader("Historial de clasificaciones")
    st.dataframe({
        "Mensaje": st.session_state.messages[::-1],
        "Clasificación": st.session_state.tags[::-1],
    })

if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        raise RuntimeError("No se pudo cargar el modelo o el vectorizador.")

    app(model, vectorizer)
