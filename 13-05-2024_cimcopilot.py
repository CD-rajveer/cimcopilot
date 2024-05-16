# Standard library imports
import base64
from io import BytesIO
import os
import json
import re
import sqlite3
import time
import joblib
import pandas as pd
import requests
from streamlit_text_rating.st_text_rater import st_text_rater
from streamlit_feedback import streamlit_feedback
import logging
import speech_recognition as sr
import tempfile
import csv
from  sessionState import get



# Third-party imports
import streamlit as st
from streamlit import session_state
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory


from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
)

google_api_key = "AIzaSyDYmei4WbkEX9p_6rF_RaANkl72DgxEIBQ"
SERPER_API_KEY = "01003b150933b133d45925dc3c532bef4309c00e"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API _keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# data_path = "./CIMdata"



store = {}



loaders = {
    ".pdf": PyMuPDFLoader,
    ".xml": UnstructuredXMLLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def fetch_files_from_folder(folder_path):
    pdf_files = []
    doc_files = []

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            pdf_files.append(file_path)
        elif filename.endswith(".doc") or filename.endswith(".docx"):
            doc_files.append(file_path)

    return pdf_files, doc_files


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if not pdf:  # Check if pdf is empty
            continue  # Skip to next iteration if empty
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    # chunks = text_splitter.split_documents(text)
    return chunks


@st.cache_data(show_spinner=False)
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )
    vector_store = FAISS.from_texts(
        text_chunks, embedding=embeddings
    )
    vector_store.save_local("cimcopilot_vector_data")


###########################################################################################################
##############################               ADD REMEMBER CONVERSATION_MEMEORY ############################
###########################################################################################################

chat_message_history = ChatMessageHistory()
# history = StreamlitChatMessageHistory(key="chat_messages")

def handle_user_input_and_ai_response(user_question, ai_response):
    # Add user's question to the chat history
    chat_message_history.add_user_message(user_question)
   
    # Add AI's response to the chat history
    chat_message_history.add_ai_message(ai_response)
   


# def get_conversational_chain():

#     groq_chat = ChatGroq(
#             groq_api_key="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",
#             model_name="llama3-70b-8192",
#             temperature=0.1,
#             max_tokens=256
#     )
 
#     contextualize_q_system_prompt = """Given a chat history and the latest user question \
#     which might reference context in the chat history, formulate a standalone question \
#     Do not create wrong question.
#     which can be understood without the chat history. Do NOT answer the question, \
#     just reformulate it if needed and otherwise return it as is."""
 
#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#     history_aware_retriever = create_history_aware_retriever(
#         groq_chat, db.as_retriever(search_kwargs={"k": 6}), contextualize_q_prompt
#     )
#     #"gemma-7b-it"#"llama2-70b-4096"

#     model = ChatGoogleGenerativeAI(
#         model="gemini-1.5-pro-latest",
#         temperature=0.2,
#         max_output_tokens=2048,
#         verbose=True,
#         google_api_key=google_api_key,
#     )
 
#     sys_prompt = """You are a very helpful, expert, and supportive Q&A Assistant called CIMCON Sarathi Bot.
#     Your task is to understand the question and requirements and then provide clear and detailed answers by understanding the provided context.
#     Before answering the question make sure that answer is correct and according to provided context, But Do not provide wrong or false information you can verify answer with context before provide. Give answers in a proper format.
#     When responding, ensure to include relevant links from the context if available. If you find any relevant links from the context, it's mandatory to include them in your response.
#     When answering, add supportive text and give the answer in a friendly manner.
#     If the query is a greeting or normal talk, reply accordingly.
#     If you find a spelling mistake in the question, utilize fuzzy matching to identify similar terms in the question.
   
#     **You can say "I don't have information about this, Please provide relevant and clear question! or Contact Support Team on this email :- support@cimcondigital.com" unless the question is completely unrelated.**
   
#      Context:
#      {context}"""
   
#     qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", sys_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
#     )
#     question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
 
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
 
 
#     ### Statefully manage chat history ###
 
#     # def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     #     if session_id not in store:
#     #         store[session_id] = ChatMessageHistory()
#     #     return store[session_id]
 
 
#     conversational_rag_chain = RunnableWithMessageHistory(
#         rag_chain,
#         # get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer",
#     )
 
#     # answer = conversational_rag_chain.invoke({"input":question},config={
#     #     "configurable": {"session_id": id}
#     # },)
 
#     # if id in store and len(store[id].messages) > 6:
#     #     del store['abc13'].messages[:4]
 
#     answer = answer['answer']
    



#     load_qa_prompt_template = """
#     You are helpful conversational assistant CIMCopilot created by **CIMCON Digital** for configuration assistance of edge devices with cim10. answer should be from the {context} Respond in a 
# friendly and helpful tone, with precise and concise answer. make sure to ask user relevant follow-up questions.generate configuration json if user requests.
 
# As an adept in analyzing technical specifications and configurations for a broad array of edge devices, 
# you are  here to offer precise and detailed information regarding edge device specifications and its configuration with CIM10, in response
# to specific queries. your proficiency spans across various manufacturers, encompassing crucial details such as make, model number, power consumption, processing capabilities, connectivity options, input/output interfaces, operating range, and firmware/software support.

# Your primary task is to ask me about edge device specifications, and yo'll provide you with the relevant information. Additionally, you are equipped to provide information about the CIM10 upon request. If the requested data is unavailable directly from the context, you'll respond with "answer is not available in the context." . For more detailed inquiries, you can contact our support team at support@cimcondigital.com.

# ---

# **Regarding CIM10 Analog Input Configuration:**

# set measuring range value from according to pipe diameter extract useful measuring in Scale low and scale high 


# you need to generate this below yml as json
# ai_config:
#   publisher:
#     destination: []
#     sampling_rate: 60
#     debug: 0
#   aiChannel:
#     - Enable: 1
#       pinno: 1
#       ChannelType: I
#       EnggLowCal: 4
#       EnggHighCal: 20
#       Scalelow: 0
#       Scalehigh: 100
#       Name: add
#       peripheral_id: "1234567891234567899"
#       uuid: "6b46bd14-061d-11ef-b228-60b6e10ad793"


# For configuring analog input on the CIM10, there are two pins available, supporting up to two analog channels at maximum. Various parameters need configuration such as Range, Destination, Sampling Rate, Debug, NumOfChannels, Enable, Pin Number, Channel Type, Name, Peripheral ID, EnggLowCal, EnggHighCal, Scalelow, and Scalehigh. These parameters facilitate effective calibration, configuration, and conversion of raw and engineering values.

# For instance, if configuring a field device with a 4-20mA output on AI pin, the configuration settings on CIM10 would include:
# | Parameter           | Value                                    |
# |---------------------|------------------------------------------|
# | Pin Number          | 1                                        |
# | Sampling rate (Sec) | 1 to 86400                               |
# | Destination         | CIMCON Cloud                             |
# | Name                | User-defined                             |
# | Device ID           | User-defined                             |
# | Channel Type        | Current (I)                              |
# | Engg. Scale Low     | 4                                        |
# | Engg. Scale High    | 20                                       |
# | Scale Low           | read the and fill here   |
# | Scale High          | read the and fill here   |

# Or, if the channel is a voltage channel:

# | Parameter           | Value                                    |
# |---------------------|------------------------------------------|
# | Pin Number          | 1                                        |
# | Sampling rate (Sec) | 1 to 86400                               |
# | Destination         | CIMCON Cloud                             |
# | Name                | User-defined                             |
# | Device ID           | User-defined                             |
# | Channel Type        | Voltage (V)                              |
# | Engg. Scale Low     | 0                                        |
# | Engg. Scale High    | 10                                       |
# | Scale Low           | Read from datasheet and fill here        |
# | Scale High          | Read from datasheet and fill here        |

# ---

# **Regarding CIM10 Digital Input Configuration:**

# you need to generate this xml as json format or table format


# <di_config>
#   <publisher>
#     <destination/>
#     <sampling_rate>60</sampling_rate>
#     <debug>0</debug>
#   </publisher>
#   <DiChannel>
#     <item>
#       <pin_no>1</pin_no>
#       <pin_name>hello</pin_name>
#       <peripheral_id>123456789123456789</peripheral_id>
#       <uuid>8e37d75f-faf5-11ee-88ce-60b6e10ad793</uuid>
#     </item>
#   </DiChannel>
# </di_config>

# while generating json or table remove item from the xml 

# For configuring digital input on the CIM10, there are two pins available, supporting up to two digital channels at maximum. Parameters such as range, Sampling Rate, Debug, Number Of Channel, and Device ID need configuration.

# | Parameter           | Value                                    |
# |---------------------|------------------------------------------|
# | Pin Number          | 1                                        |
# | Sampling rate (Sec) | 1 to 86400                               |
# | Destination         | CIMCON Cloud                             |
# | Pin Name            | User-defined                             |
# | Device ID           | User-defined                             |


# if you do not have answer **You can say "I don't have information about this, Please provide relavent and clear question! or Contact Support Team on this email :- support@cimcondigital.com" unless the question is completely unrelated.*
# remember the conversation in the chat.

# ---
# Context: 
# chat history: 
# \n{chat_history}\n
# Question: 
# \n{question}\n 

# Chatbot:
#     """

#     load_qa_prompt = PromptTemplate(
#         template=load_qa_prompt_template,
#         input_variables=["context", 
#                         #  "datasheet_text", 
#                          "chat_history",
#                          "question"],
#     )

#     summary_prompt_template = """
#             Current summary:
#             {summary}

#             new lines of conversation:
#             {new_lines}

#             New summary:
#             """
    
#     SUMMARY_PROMPT = PromptTemplate(input_variables=["summary", "new_lines"], template=summary_prompt_template)
    

#     # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_output_tokens=4096)
#     # master_prompt = prompt_template + json_structure
    

#     memory = ConversationBufferMemory(
#         llm=model,
#         memory_key="chat_history",
#         input_key="question",
#         max_token=5000, # after 5000 token, summary of the conversation will be created and stored in moving_summary_buffer
#         prompt=SUMMARY_PROMPT,
#         moving_summary_buffer="summary", # this sets the summary of the memory
#         chat_memory=chat_message_history, # this sets the previous conversation chat history
#     )

# #     memory = ConversationBufferMemory(
# #     chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
# #     return_messages=True, # Used to use message formats with the chat model
# #     memory_key="chat_history",
# # )
    
    
#     chain = load_qa_chain(model, chain_type="stuff", prompt=load_qa_prompt, memory=memory)
#     # print(memory)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         google_api_key=google_api_key,
#     )
#     doc_db = FAISS.load_local(
#         "cimcopilot_vector_data",
#         embeddings,
#         allow_dangerous_deserialization=True,
#     )
#     # table_db = FAISS.load_local(
#     #     "table_vector_Data",
#     #     embeddings,
#     #     allow_dangerous_deserialization=True,
#     # )
#     docs_retriever = doc_db.similarity_search(user_question, 4)
#     # table_retriever = table_db.similarity_search(user_question, 4)
#     # tables = []
#     # for doc in table_retriever:
#     #     try:
#     #         # if score <= 0.4:
#     #         if doc.metadata["type"] == "table":
#     #             tables.append(doc.metadata["original_content"])
#     #     except:
#     #         pass
#     # tables
#     chain = get_conversational_chain()
#     response = chain.invoke(
#         {
#             "input_documents": docs_retriever,
#             # "datasheet_text": tables,
#             "question": user_question,
#         },
#         return_only_outputs=True,
#     )
#     data = {
#         "user_question": user_question,
#         "response": response["output_text"],
#     }
#     with open("user_question_response.json", "w") as json_file:
#         json.dump(data, json_file)

#     # Save user question and response in text format
#     with open("user_question_response.txt", "a") as txt_file:
#         txt_file.write(f"User Question: {user_question}\n")
#         txt_file.write(f"Response: {response['output_text']}\n")
#         txt_file.write("\n")

#     return response["output_text"]

def store_question_response_feedback(user_question, response, feedback):
    with open("user_question_response_feedback.txt", "a") as txt_file:
        txt_file.write(f"User Question: {user_question}\n")
        txt_file.write(f"Response: {response}\n")
        txt_file.write(f"Feedback: {feedback}\n")
        txt_file.write("\n")

def print_fdbk():
    print(st.session_state.fbk)
    st.session_state.qry = ""

def create_feedback():
    with st.form('feedback_form'):
        user_feedback = st.text_input("Please provide your feedback: ")
        submitted_input = st.form_submit_button('Submit question', on_click = create_feedback)



# Setting up logging configuration
# logging.basicConfig(level=logging.INFO)

def user_input(user_question, chat_history):
# def user_input(user_question):
    corrected_chat_history = []
    for msg in chat_history:
        if 'human' in msg:
            corrected_chat_history.append(('human', msg['human']))
        elif 'ai' in msg:
            corrected_chat_history.append(('ai', msg['ai']))
        else:
            raise ValueError("Unexpected message type. Use 'human' or 'ai'.")

    # logging.info(f"Received user question: {user_question}")
    # chat_history_tuples = [(msg["human"], msg["ai"]) for msg in chat_history]
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key,
    )

    doc_db = FAISS.load_local(
        "cimcopilot_vector_data",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    # logging.info("Loaded document database.")

    groq_chat = ChatGroq(
            groq_api_key="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",
            model_name="llama3-70b-8192",
            temperature=0.1,
            max_tokens=256
    )
    # logging.info("Initialized GROQ chat.")

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    Do not create wrong question.
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    logging.info("Prepared contextualize question system prompt.")

    load_qa_prompt_template = """
    You are helpful conversational assistant CIMCopilot created by **CIMCON Digital**, dedicated to providing accurate,
    clear, and detailed answers from the provided context and conversation history.
    If the query is a greeting or normal talk, reply accordingly.
    for other queries, answer should be from the {context}. Respond in a friendly and helpful tone, with precise and concise answer. 
    Regarding configuration assistance of field devices with cim10, reply accurately. make sure to ask user relevant follow-up questions. generate configuration json if user requests.
    You need to explain the answer instead of just copying and pasting text from the context and always base your answer on the information found within the context.
    please provide answer specific to the field device mentioned.
    remember the this chat history {chat_history} asked and act accordingly. 
    Use fuzzy matching to correct spelling mistakes in the question and identify similar terms. 

    you need to generate this below yml as json if asked to generate for AI configuration.
ai_config:
  publisher:
    destination: []
    sampling_rate: 60
    debug: 0
  aiChannel:
    - Enable: 1
      pinno: 1
      ChannelType: I
      EnggLowCal: 4
      EnggHighCal: 20
      Scalelow: 0
      Scalehigh: 100
      Name: add
      peripheral_id: "1234567891234567899"
      uuid: "6b46bd14-061d-11ef-b228-60b6e10ad793"

      or if asked for digital input generate following as an 
      <di_config>
          <publisher>
            <destination/>
            <sampling_rate>60 by default</sampling_rate>
            <debug>0 or 1</debug>
          </publisher>
          <DiChannel>
            <item>
              <pin_no>1 or 2</pin_no>
              <pin_name>device name</pin_name>
              <peripheral_id>123456789123456789(dummy) can be changed</peripheral_id>
              <uuid>8e37d75f-faf5-11ee-88ce-60b6e10ad793</uuid>
            </item>
          </DiChannel>
    </di_config>

        generate json for device configuration if requested:
export_device:
  device_config:
    - devicename: test1
      protocol: modbusrtu
      address: ""
  configlist:
    Test1:
      version: "1.0.0"
      publisher:
        destination: []
        protocol: modbus-serial
        sample_rate: 1
        debug: 0
      devicename: meter1
      enable: 0
      serial_port:
        device: "/dev/ttyMAX0"
        baud_rate: 9600
        parity: "N"
        data_bits: 8
        stop_bits: 1
        xon_off: 0
        port_timeout: 3000
      query: []

      generate json for device configuration if asked

    you also need to generate python scipt for the indstrial use cases for automation and iot.
    if the answer cannot be found, respond with a simple statement: "I don't have information about this, 
    Please provide relavent and clear question! "I don't have information about this, Please provide relavent and
    clear question! or Contact Support Team on this email :- support@cimcondigital.com" Do not include any additional explanations or details beyond this statement.\n\n
    **You can say "I don't have information about this, Please provide relavent and clear question! or
    Contact Support Team on this email :- support@cimcondigital.com" unless the question is completely unrelated.*
    """
    # logging.info("Prepared load QA prompt template.")

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            *corrected_chat_history, 
            ("human", "{input}"),
        ]
    )
    # logging.info("Prepared contextualize question prompt.")

    history_aware_retriever = create_history_aware_retriever(
        groq_chat, doc_db.as_retriever(search_kwargs={"k": 7}), contextualize_q_prompt
    )
    # logging.info("Created history-aware retriever.")

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.2,
        max_output_tokens=2048,
        verbose=True,
        google_api_key=google_api_key,
        convert_system_message_to_human=True,
        # safety_settings=
    )
    # logging.info("Initialized Google generative AI model.")

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_qa_prompt_template),
            MessagesPlaceholder("chat_history"),
            *corrected_chat_history,
            ("human", "Hello, How are you doing?"),
            ("ai", "Hello, I am doing well."),
            ("human", "{input}"),
        ]
    )
    # logging.info("Prepared QA prompt.")

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    # logging.info("Created question-answer chain.")

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # logging.info("Created retrieval chain.")

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    # logging.info("Defined function to get session history.")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    # logging.info("Created conversational RAG chain.")

    response  = conversational_rag_chain.invoke({"input":user_question},config={
        "configurable": {"session_id": id}
    },)
    # logging.info("Invoked conversational RAG chain.")

    data = {
        "user_question": user_question,
        "response": response["answer"],
    }
    with open("user_question_response.json", "w") as json_file:
        json.dump(data, json_file)
    # logging.info("Saved response to JSON file.")
# 
    # Save user question and response in text format
    with open("user_question_response.txt", "a") as txt_file:
        txt_file.write(f"User Question: {user_question}\n")
        txt_file.write(f"Response: {response['answer']}\n")
        txt_file.write("\n")
    # logging.info("Saved response to text file.")

    return response['answer']

def download_pdf(url, directory="."):
    """
    Download a PDF file from the given URL and save it in the specified directory.

    Args:
        url (str): The URL of the PDF file to download.
        directory (str): The directory where the PDF file will be saved. Defaults to "." (current directory).

    Returns:
        str: The path to the downloaded PDF file.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Extract filename from URL
    filename = os.path.join(directory, url.split("/")[-1])

    # Download PDF file
    with open(filename, "wb") as f:
        response = requests.get(url)
        f.write(response.content)
    return filename


def download_pdfs_from_search_results(
    search_query, directory="downloaded_pdfs"
):
    """
    Search for a query using Google Serper API and download PDF files from the search results.

    Args:
        search_query (str): The search query to use for searching.
        directory (str): The directory where the downloaded PDF files will be saved. Defaults to "downloaded_pdfs".
    """
    # Initialize GoogleSerperAPIWrapper
    search = GoogleSerperAPIWrapper()

    # Perform search and get results
    results = search.results(search_query)

    # Extract PDF links from search results
    pdf_links = [
        item["link"]
        for item in results.get("organic", [])
        if item["link"].endswith(".pdf")
    ]

    # Download PDFs
    downloaded_files = []
    for link in pdf_links:
        downloaded_files.append(
            download_pdf(link, directory=directory)
        )
    return downloaded_files

# def save_chat_history_to_csv(chat_history):
#     chat_history.to_csv("chat_history.csv", index=False)

# def load_chat_history_from_csv():
#     try:
#         return pd.read_csv("chat_history.csv")
#     except FileNotFoundError:
#         return pd.DataFrame(columns=["ChatID", "role", "content"])


# Function to get button label for each chat
# def get_button_xlabel(chat_df, chat_id):
# first_message = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "user")].iloc[0]["Content"]
# return f"Chat {chat_id[0:7]}: {' '.join(first_message.split()[:5])}..."
@st.cache_data
def convert_df_to_csv(data):
    df = pd.DataFrame(data)
    return df.to_csv().encode("utf-8")


def response_generator(response):
    for word in response.split():
        yield word + " "
        # yield word 
        # word
        time.sleep(0.0005)


# def display_parse_storePDF(pdf_url):
#     # Opening file from file path
#     # Fetching PDF content from the URL
#     logging.info(f"start {pdf_url} rendering")
#     pdf_content = requests.get(pdf_url).content

#     # Converting PDF content to base64

#     base64_pdf = base64.b64encode(pdf_content).decode("utf-8")

#     # Embedding PDF in HTML
#     pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

#     # Displaying PDF
#     st.markdown(pdf_display, unsafe_allow_html=True)
#     logging.info(f"pdf  {pdf_url} rendered")

#     pdf_name = os.path.basename(pdf_url)

#     logging.info(f"start parsing pdf {pdf_url}")
#     if st.button(f"Parse '{pdf_name}'"):
#         try:
#             # Display message indicating parsing is starting
#             st.info(f"Parsing PDF '{pdf_name}'...")
#             logging.info(f"start parsing pdf {pdf_url}")
#             # Open the PDF file from the provided link
#             with requests.get(pdf_url, stream=True) as response:
#                 with open(pdf_name, "wb") as pdf_file:
#                     pdf_file.write(response.content)
#             # print(response.content)

#             # Extract text from the PDF
#             logging.info(f"parsing pdf text")
#             text = get_pdf_text(response.content)
#             logging.info(f"parsed  pdf")

#             # Get text chunks
#             get_text_chunks(text)
#             logging.info(f"got text chunks")

#             # Process text chunks or perform any further operations
#             # For example:
#             # get_vector_store(text_chunks)

#             # Clean up temporary PDF file
#             # os.remove(pdf_name)

#             # Display message indicating parsing is done
#             st.toast("PDF parsing done.")
#             logger.info(f"PDF parsing successful: {pdf_name}")
#         except Exception as e:
#             st.error(f"Error parsing PDF '{pdf_name}': {e}")
#             logger.error(f"Error parsing PDF '{pdf_name}': {e}")
#     elif st.button("Cancel"):
#         st.error("PDF parsing cancelled.")
#         logger.info("PDF parsing cancelled.")


def parse_pdf():
    datasheet_text = None
    try:
        pdf_name = (
            st.session_state.pdf_name
        )  # Get pdf_name from session state
        # Display message indicating parsing is starting
        st.toast(f"Parsing datasheet '{pdf_name}'...")

        # Save PDF content to a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp_file:
            temp_file.write(st.session_state.pdf_content)
            temp_file_path = temp_file.name

        # Extract text from the PDF
        # logging.info(f"parsing pdf text")
        datasheet_text = get_pdf_text([temp_file_path])  # Pass file path
        # logging.info(f"parsed  pdf")

        # Get text chunks
        chunks = get_text_chunks(datasheet_text)
        get_vector_store(chunks)
        # logging.info(f"got text chunks: {chunks}")

        # Display the text chunks
        # st.write("Text Chunks:")
        # for i, chunk in enumerate(chunks):
        #     st.write(f"Chunk {i+1}: {chunk}")

        # Process text chunks or perform any further operations
        # For example:
        # get_vector_store(text_chunks)

        # Display message indicating parsing is done
        st.toast("PDF parsing done.")
        # return datasheet_text

    except Exception as e:
        st.error(f"Error parsing PDF {pdf_name}")
        # return None
    finally:
        # Clean up temporary file
        if temp_file_path:
            os.unlink(temp_file_path)


def display_parse_storePDF(pdf_url):
    # Opening file from file path
    # Fetching PDF content from the URL
    logging.info(f"start {pdf_url} rendering")
    pdf_response = requests.get(pdf_url)
    pdf_content = pdf_response.content
    # pdf_content = get_pdf_text(pdf_content)
    # print(pdf_content)

    # Converting PDF content to base64
    base64_pdf = base64.b64encode(pdf_content).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying PDF
    st.markdown(pdf_display, unsafe_allow_html=True)
    logging.info(f"pdf  {pdf_url} rendered")

    # Store PDF content and name in session state
    st.session_state.pdf_content = pdf_content
    st.session_state.pdf_name = os.path.basename(pdf_url)

    # Prompt user for confirmation
    st.write("Is the datasheet correct?, Do I need to parse it?")

    # Create two columns for buttons with adjusted width
    col1, col2, col3 = st.columns([1, 1, 9])

    if col1.button(
        "Yes", key="parse_button", on_click=parse_pdf, type="primary"
    ):
        logging.info(f"start parsing pdf {pdf_url}")
    if col2.button(
        "No", key="search_button", on_click=search_datasheets
    ):
        pass
    else:
        st.info(
            "Please review the datasheet and confirm its correctness before parsing or search again."
        )


# TODO: add search on intenet feature when user responds incorrect datasheet
def search_datasheets():
    st.warning("Searching for datasheets on the internet...")
    # Trigger internet search and download process
    # search_query = st.text_input("Enter your search query:")
    # if st.button("Search"):
    # pdf_files = download_pdfs_from_search_results(search_query)
    # if pdf_files:
    # st.toast("Datasheet found and downloaded successfully!")
    # Perform further actions if needed
    # else:
    st.toast("No datasheet found. Please refine your search query.")


# def parse_table_from_response(response):
#     # Extract the table from the chatbot response
#     table_text = re.search(r"\| Parameter\s+\| Value\s+\|(.+?)\|", response, re.DOTALL)

#     if table_text:
#         table_lines = table_text.group(1).strip().split('\n')[1:]  # Exclude table header

#         # Parse the table into a list of dictionaries
#         table_data = []
#         for line in table_lines:
#             parameter, value = [item.strip() for item in line.split('|') if item.strip()]
#             table_data.append({"Parameter": parameter, "Value": value})

#         json_string = json.dumps(table_data)

#         # st.json(json_string, expanded=True)

#         st.download_button(
#             label="Download JSON",
#             file_name="data.json",
#             mime="application/json",
#             data=json_string,
#         )
#     else:
#         return None


def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )
        return ""


# def extract_json_table(text):
#     # Regular expressions to match JSON and table sections
#     json_pattern = r"""(?x)
#                 \{                   # Match opening curly brace
#                   (?:                 # Non-capturing group for content
#                     [^{}]+?           # Match any character except curly braces one or more times
#                   |                     # OR
#                     "(?:\\.|[^"\\])*?"  # Match a string literal (including escaped quotes)
#                   )*                   # Zero or more repetitions
#                 \}                   # Match closing curly brace
#               """
#     table_pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'

#     # Find JSON matches
#     # json_matches = re.findall(json_pattern, text)
#     # json_matches = re.findall(json_pattern, text)
#     json_data = {}
#     pattern = re.compile(json_pattern, flags=re.DOTALL)  # Allow matching newlines
#     json_matches = pattern.findall(text)

#     # Convert each JSON match to JSON object
#     for match in json_matches:
#         try:
#             json_obj = json.loads(match)
#             if 'ai_config' in json_obj:
#                 json_data['ai_config'] = json_obj['ai_config']
#             elif 'di_config' in json_obj:
#                 json_data['di_config'] = json_obj['di_config']
#         except json.JSONDecodeError:
#             pass

#     table_matches = re.findall(table_pattern, text)
#     table_data = []

#     # Convert table matches to JSON objects
#     for match in table_matches:
#         table_data.append({match[0]: match[1]})

#     return json_data
# def extract_json_table(text):
#     json_pattern = r'```json\s*(.*?)```'

#     pattern = re.compile(json_pattern, flags=re.DOTALL)
#     json_matches = pattern.findall(text)
#     json_matches_no_comments = [re.sub(r'\/\/.*$', '', match, flags=re.MULTILINE) for match in json_matches]
#     # json_matches_no_comments = [match.replace('\n', '') for match in json_matches_no_comments]

#     return json_matches_no_comments


def extract_json_table(text):
    json_pattern = r"```json\s*(.*?)```"
    table_pattern = r"```(.*?)```"

    json_matches = re.findall(json_pattern, text, flags=re.DOTALL)
    json_matches_no_comments = [
        re.sub(r"\/\/.*$", "", match, flags=re.MULTILINE)
        for match in json_matches
    ]

    table_matches = re.findall(table_pattern, text, flags=re.DOTALL)
    if table_matches:
        for match in table_matches:
            if "json" in match:
                json_table = re.findall(json_pattern, match)
                json_matches_no_comments.extend(json_table)
    elif json_matches:
        json_matches_no_comments.extend(json_matches)

    return json_matches_no_comments


# def download_json(json_data, config_type):
#     filename = f"{config_type}.json"
#     with open(filename, "w") as file:
#         json.dump(json_data, file, indent=4)


# def handle_feedback():
#     st.write(st.session_state.fb_k)
#     st.toast("✔️ Feedback received!")

def create_database():
    os.makedirs("database/user", exist_ok=True)
    conn = sqlite3.connect('database/user/user.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, response TEXT, feedback TEXT)''')
    conn.commit()
    conn.close()

# Function to insert data into the database
def insert_feedback(question, response, feedback):
    conn = sqlite3.connect('database/user/user.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback (question, response, feedback) VALUES (?, ?, ?)",
              (question, response, feedback))
    conn.commit()
    conn.close()


def main():
    st.set_page_config("CIMCOPILOT")
    st.header("CIMCOPILOT")
    st.text("Start your chat")
    cimcopilot = """Hello there! I'm CIMCopilot, your intelligent digital assistant here to make your life easier when setting up your CIM10 IoT Edge gateway. Using cutting-edge AI, ML, and NLP technologies, I'm here to guide you through every step of the process, no matter your technical background.\

        \n**Setup Made Easy:**
        With my intuitive chat interface, I can understand your questions and provide clear answers. Need help with physical device connections? No problem, I've got you covered. I can even generate and help you download the configuration files you need.\

        \n**Seamless Configuration:**
        Uploading configurations to your CIM10 devices? Piece of cake! I'll walk you through the process step by step, ensuring everything runs smoothly.\

        \n**Continuous Improvement:**
        I'm not just a one-time helper. I'm always learning from your interactions to provide even better assistance in the future. Think of me as your personal IoT guru, dedicated to enhancing your experience with the CIM10 gateway.\

        \n**User-Centric Support:**
        My goal is to minimize downtime and maximize your satisfaction. With CIMCopilot by your side, managing your IoT devices has never been more seamless.\
        """
        # with st.chat_message("ai"):
            # st.markdown(cimcopilot)
    session_state_1 = get(cimcopilot_message=cimcopilot)

# Display the message
    st.write(session_state_1.instance.cimcopilot_message)

    # memory = ChatMessageHistory()
    memory  = ConversationBufferMemory()

    # Folder path where PDF and DOC files are stored
    folder_path = "./CIMdata"

    # Fetch PDF and DOC files from the folder
    pdf_files, doc_files = fetch_files_from_folder(folder_path)

    # if not pdf_files:
    #     st.warning("No PDF files uploaded. Performing internet search for datasheets...")
    #     # Trigger internet search and download process
    #     search_query = st.text_input("Enter your search query:")
    #     if st.button("Search"):
    #         pdf_files = download_pdfs_from_search_results(search_query)

    # Function to extract text from PDF files
    pdf_texts = []
    for pdf_file in pdf_files:
        pdf_texts.append(get_pdf_text([pdf_file]))

    # Get text chunks
    text_chunks = get_text_chunks(" ".join(pdf_texts))

    # Generate vector stst.session_state.messages.append({"role": "assistant", "content": response})ore
    get_vector_store(text_chunks)

    if "chat_history" not in st.session_state:
            st.session_state.chat_history= []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output':message['ai']})

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # Display chat messages from history on app rerun
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # new_chat_id = f'{time.time()}'
    MODEL_ROLE = "ai"
    AI_AVATAR_ICON = "✨"

    current_time = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()
    )
    new_chat_id = f"{current_time}"

    try:
        past_chats: dict = joblib.load("data/past_chats_list")
    except:
        past_chats = {}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    def create_new_chat(chat_id):
        st.session_state["chat_history"][chat_id] = []

    def add_message_to_chat(chat_id, message, role):
        if chat_id not in st.session_state["chat_history"]:
            create_new_chat(chat_id)
        st.session_state["chat_history"][chat_id].append(
            {"role": role, "content": message}
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history"
        )

    with st.sidebar:
        image = "./images/CIMcopilot Logo-01.png"
        st.image(image, use_column_width=True)
        toggle_state = st.toggle("Find datasheet online")
        st.title("Upload datasheet")
        pdf_docs = st.file_uploader(
            "Upload your datasheets and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        st.write("# Past Chats")
        if st.session_state.get("chat_id") is None:
            st.session_state.chat_id = st.selectbox(
                label="Pick a past chat",
                options=[new_chat_id] + list(past_chats.keys()),
                format_func=lambda x: past_chats.get(x, "New Chat"),
                placeholder="_",
            )
        else:
            st.session_state.chat_id = st.selectbox(
                label="Pick a past chat",
                options=[new_chat_id, st.session_state.chat_id]
                + list(past_chats.keys()),
                index=1,
                format_func=lambda x: past_chats.get(
                    x,
                    (
                        "New Chat"
                        if x != st.session_state.chat_id
                        else st.session_state.chat_title
                    ),
                ),
                placeholder="_",
            )
        st.session_state.chat_title = (
            f"CIMCopilot-{st.session_state.chat_id}"
        )

    # Check if the flag file exists
    

    # """##################################Add Voice InPUT#####################################"""
    # voice_text = ''
    # if st.button('Record'):
    #     with st.spinner(f'Recording..'):
    #         voice_text = get_voice_input()
    #         print(voice_text)
    #     st.success("Recording completed")
    # st.text_input("Text Input", value=voice_text)
    ###################################################################################

    # Input box for custom question
    # custom_question = st.text_area("Or type your own question here:", max_chars=5000, value = selected_text)

    # Get user question based on selection or custom input
    # user_question = selected_text if selected_text else custom_question
    user_question = st.chat_input(
        "Message CIMCopilot", max_chars=5000, disabled=False
    )

    # if st.button("Start Speech Input", on_click=transcribe_speech()):
    #     speech_text = transcribe_speech()
    #     if speech_text:
    #         user_question = speech_text

    if user_question:
        if toggle_state:
            api_key = os.getenv("SERPER_API_KEY")
            # api_key = "01003b150933b133d45925dc3c532bef4309c00e"
            st.toast("Finding datasheet online!")
            search = GoogleSerperAPIWrapper()
            results = search.results(user_question)
            files_and_links = [
                (result["title"], result["link"])
                for result in results["organic"]
            ]
            response = None
            pdf_link = None
            for file_name, link in files_and_links:
                response = f"[{file_name}]({link})"
                if link.endswith((".pdf", ".docx", ".book")):
                    pdf_link = link
                    break

            # user_input(user_question)
            st.session_state.messages.append(
                {"role": "user", "content": user_question}
            )
            # st.write(f"User: {user_question}")
            with st.chat_message("user"):
                st.markdown(user_question)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            with st.chat_message("assistant"):
                with st.status("Downloading data..."):
                    st.write("Searching for data...")
                    time.sleep(2)
                    st.write("Found URL.")
                    time.sleep(1)
                    st.write("Downloading data...")
                    time.sleep(1)
                # st.write(response)
                # ratings = st_text_rater(text=st.markdown(response))
                st.write_stream(response_generator("I found below datasheet online on your request" + response))
                # st.write(response)
                if pdf_link:
                    # Display PDF using displayPDF function
                    display_parse_storePDF(pdf_link)
                # text = get_pdf_text(pdf_link)
                # get_text_chunks(text)
                # st.download_button(label="Download Config", data=json_data, file_name='ai_config.json', mime='application/json')
            # add_message_to_chat(
            #     st.session_state.chat_id, user_question, "user"
            # )
            # # Add assistant response to chat history
            # add_message_to_chat(
            #     st.session_state.chat_id, response, "assistant"
            # )
            # Add the user's question to the chat history

        else:
            response = user_input(user_question, st.session_state.chat_history)
            # response = user_input(user_question)
            message = {'human':user_question, 'ai': response}
            # print("Before appending:", st.session_state.chat_history)
            if not isinstance(st.session_state.chat_history, list):
                st.session_state.chat_history = []
            # else:
                # print("chat_history is not a list:", st.session_state.chat_history)
            # st.session_state.chat_history.append(message)
            st.session_state.chat_history.append(message)

            # user_input(user_question)
            st.session_state.messages.append(
                {"role": "user", "content": user_question}

            )
            
            # st.write(f"User: {user_question}")
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            
            with st.chat_message("assistant"):
                # st.write(response)
                # ratings = st_text_rater(text=st.markdown(response))
                # st.write(response_generator(response))
                st.write(response)  
                # st.markdown(response)
            # with st.expander(label = "chat_history", expanded=False):
            #     st.write(st.session_state.chat_history)
            #     st.download_button("chat download", json.dumps(st.session_state.chat_history, indent=4), file_name="history.json", mime="application/json")
             
                
                # create_feedback()
            # feedback = streamlit_feedback(
            #     feedback_type="thumbs",
            #     optional_text_label="[Optional] Please provide an explanation",
            # )
            # # st.write(feedback)
            # # Check if feedback is provided
            # if feedback:
            #     # Append feedback to messages
            #     st.session_state.messages.append({"role": "user", "content": feedback})
            #     with st.chat_message("user"):
            #         st.markdown(f"[Feedback]: {feedback}")
            # else:
            #     st.write("No feedback received")
            
            ########################## function to download_json data####################
            json_data = extract_json_table(response)
            if json_data:
                json_str = json.dumps(json_data[0])
                json_obj = json.loads(json_str)
                st.download_button(
                    label="Download JSON",
                    data=json_obj,
                    file_name="ai_config.json",
                    mime="application/json",
                )

            # config_type = list(json_data.keys())[0] if json_data else None
            # json_str = json.dumps(json_data[config_type], indent=2)

            # if config_type:
            # feedback = streamlit_feedback(
            #     feedback_type="thumbs",
            #     optional_text_label="[Optional] Please provide an explanation",
            # )
            feedback = st.text_input("Add feedback here")
            create_database() 
            insert_feedback(user_question, response, feedback)

            # if feedback:
                  
                # insert_feedback(user_question, response, json.dumps(feedback))

            # Print confirmation message
            if feedback:
                print("Feedback saved successfully.")
            else:
                print("No feedback received.")
                
            
            # # Save DataFrame to CSV
            # add_message_to_chat(
            #     st.session_state.chat_id, user_question, "user"
            # )
            # # Add assistant response to chat history
            # add_message_to_chat(
            #     st.session_state.chat_id, response, "assistant"
            # )
    with st.sidebar:
        with st.expander(label = "chat_history", expanded=False):
            st.write(st.session_state.chat_history)
            st.download_button("chat download", json.dumps(st.session_state.chat_history, indent=4), file_name="history.json", mime="application/json")


if __name__ == "__main__":
    main()
