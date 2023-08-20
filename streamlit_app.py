import streamlit as st
from streamlit_option_menu import option_menu
from main import *
from streamlit_chat import message

st.set_page_config(layout='wide')

html_temp = """
    <div style="background-color:tomato;padding:3px">
    <h2 style="color:white;text-align:center;padding:3px">Doc-Chat</h2>
    <h4 style="color:white;text-align:center;padding:3px">Talk with your documents</h4>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

def restart(): 
    st.session_state.page = 0

with st.sidebar:
    page = option_menu(
        menu_title="Index",
        options = ['Upload Documents', 'Chat Section']
    )

if page == 'Upload Documents':
    st.subheader(':blue[Upload your files: ]')
    uploaded_file = st.file_uploader('', accept_multiple_files=True)

    if uploaded_file is not None: 
        with st.spinner('Please wait while we do the embeddings...'):        
            st.write(create_embeddings(uploaded_file))

if page == 'Chat Section':
    user_question = st.chat_input("Put your question here..")
    if user_question:
        st.write(handle_userinput(user_question))