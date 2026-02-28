import streamlit as st
import pickle

model = pickle.load(open('reel_model.pkl','rb'))
le = pickle.load(open('encoder.pkl','rb'))

st.set_page_config(page_title="Spectra AI", page_icon="🌈")

st.title("🌈 Spectra AI")
st.subheader("Predict Your Next Reel Topic")

t1 = st.text_input("Topic 1")
t2 = st.text_input("Topic 2")
t3 = st.text_input("Topic 3")
t4 = st.text_input("Topic 4")
t5 = st.text_input("Topic 5")

if st.button("Generate Suggestion"):
    topics = [t1,t2,t3,t4,t5]
    try:
        encoded = [le.transform([t])[0] for t in topics]
        prediction = model.predict([encoded])[0]
        topic = le.inverse_transform([prediction])[0]
        st.success("🎯 Suggested Next Topic: " + topic)
    except:
        st.error("One or more topics were not seen during training.")
