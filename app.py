import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_pipeline():
    return pipeline("document-question-answering", model="impira/layoutlm-document-qa")

analyzer = load_pipeline()
st.markdown("<h1 style='text-align: left; color: #4CAF50; font-family: 'Noto Sans', sans-serif; font-size:50px;'>Intelligent Document Analyzer</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.markdown("""
        This app uses **Hugging Face Transformers** and **LayoutLM** to extract answers from documents.

        - Upload an image of a document
        - Ask a question about it
        - Get instant answers!
    """)
image=st.file_uploader("Upload the Document",type=['jpeg','jpg','png'])


if image is not None:
    try:
        uploaded_image=Image.open(image)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        question=st.text_area(label="Ask your query")
        if st.button("Submit"):
                with st.spinner("Analyzing..."):
                    data=analyzer(question=question,image=uploaded_image)
                    answer=data[0]['answer']
                    st.success(f"**Answer:** {answer}")
    except Exception as e:
        st.error("An error occurred during analysis...")