import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.title("IMDB Movie Review Sentiment Analysis")

st.write("Bu uygulama, girilen film yorumunun pozitif mi negatif mi olduğunu tahmin eder.")


model = tf.keras.models.load_model("imdb_lstm_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


maxlen = 200


yorum = st.text_area("Film yorumunu yazınız:")


if st.button("Tahmin Et"):
    
    if yorum.strip() == "":
        st.warning("Lütfen bir yorum yazınız.")
        
    else:
        yorum_dizi = tokenizer.texts_to_sequences([yorum])
        
        yorum_pad = pad_sequences(
            yorum_dizi,
            maxlen=maxlen
        )
        
        tahmin = model.predict(yorum_pad)[0][0]
        
        if tahmin >= 0.5:
            st.success("Tahmin: Pozitif Yorum")
        else:
            st.error("Tahmin: Negatif Yorum")
        
        st.write("Pozitif olma olasılığı:", round(float(tahmin), 4))