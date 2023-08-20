from dotenv import load_dotenv
import streamlit as st
import chromadb
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer
import os
from langchain.document_loaders import PDFPlumberLoader

load_dotenv()

#Hugging Face config
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('API_KEY')
repo_id = "google/flan-t5-large"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature':0.1, 'max_length':64})

#Load the documents
def load_doc(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents 

#Convert the documents into equal sized chunks
def split_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 0
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

#Convert the uploaded documents into embeddings and store in the ChromDB Database
def create_vectordb(chunks, collection_name='my_collection'):
    persist_dir = 'embeddings'
    persistent_client = chromadb.PersistentClient(path=persist_dir)
    collection = persistent_client.get_or_create_collection(collection_name)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents = chunks,
        client=persistent_client,
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=persist_dir)

    vectordb.persist()
    return vectordb

#If the embeddings already exist, Load them using this function
def load_vectordb(collection_name='my_collection'):
    persist_dir = 'embeddings'
    persistent_client = chromadb.PersistentClient(path=persist_dir)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma(client = persistent_client, collection_name=collection_name, embedding_function = embeddings)
    return vectordb

#Main chain for running the model
def chat_model(vectordb, search_kwargs={'k':1}):
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        verbose=True
    )
    return chain

#Function to create embeddings after reading documents in streamllit
def create_embeddings(uploaded_file):
    try:
        directory_path = './docs/'
        file_list = os.listdir(directory_path)
        file_list_string="\n"
        for f in uploaded_file:
            file_contents = f.read()
            file_path = os.path.join(directory_path,f.name)
            if(os.path.isfile(file_path)):
                print("file already exist no need to save !!!")
                st.warning('File already exists!!!', icon="⚠️")  
            else:
                with open(file_path, "wb") as file:
                    file.write(f.getbuffer())
                    st.sidebar.success(f"File saved", icon="✅")
                    documents = load_doc(file_path)
                    chunks = split_into_chunks(documents)
                    create_vectordb(chunks)
                    st.write("Embeddings done!!")
    except Exception as e:
        return e

#Clear the conversation
def reset_conversation():
    st.session_state.messages.clear()
    st.session_state.chat_history.clear()
    st.session_state.conversation = None

#Read the messages and present the Output
def handle_userinput(user_question):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    vectordb = load_vectordb()
    st.session_state.conversation = chat_model(vectordb)
    response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.chat_history})

    st.session_state.messages.append({'role':'User', 'content':user_question})
    st.session_state.chat_history = [tuple(response['answer'])] 
    st.session_state.messages.append({'role':'assistant', 'content':response['answer']})

    for message in st.session_state.messages: 
        with st.chat_message(message['role']):
            st.write(message['content'])

    with st.sidebar:
        st.button('Reset Conversation', on_click=reset_conversation)
