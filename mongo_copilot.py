# Standard library imports
import base64
import datetime
from io import BytesIO
import os
import json
import re
import sqlite3
import time
import uuid
import joblib
import pandas as pd
import requests
from streamlit_text_rating.st_text_rater import st_text_rater
from streamlit_feedback import streamlit_feedback
import logging
import speech_recognition as sr
import tempfile
import csv
from sessionState import get

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
    HarmBlockThreshold,
    HarmCategory,
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
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
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


####################################################################################################
####################################################################################################
########################LOG IN SECTION #############################################################
####################################################################################################
import sqlite3
from sqlite3 import Error
import os
import streamlit as st

# DATABASE_FILE = "database/users/user.db"


import os
import re
import uuid
from pymongo import MongoClient
# from pymongo.errors import ConnectionError, DuplicateKeyError
from pymongo.errors  import ConnectionFailure, DuplicateKeyError
from pymongo.server_api import ServerApi

DATABASE_URI = "mongodb+srv://cimcon:cimcon123@cluster0.x0bv6uo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "cimcopilot"
COLLECTION_NAME = "users"
CHAT_HISTORY_COLLECTION = "chat_history"

def create_connection():
    try:
        client = MongoClient(DATABASE_URI)
        db = client[DATABASE_NAME]
        return db
    except ConnectionFailure as e:
        print(e)
        return None
    


def create_collection(db):
    try:
        db.create_collection(COLLECTION_NAME)
    except Exception as e:
        print(e)

def insert_user(db, user_data):
    collection = db[COLLECTION_NAME]
    user_data["_id"] = str(uuid.uuid4())
    try:
        collection.insert_one(user_data)
        return True
    except DuplicateKeyError:
        return "Email already exists."
    except Exception as e:
        return str(e)

def authenticate_user(db, email, password):
    collection = db[COLLECTION_NAME]
    user = collection.find_one({"email": email, "password": password})
    if user:
        return user["_id"], user
    else:
        return None, None

def signup(first_name, last_name, mobile_number, email, company_name, designation, password):
    db = create_connection()
    if db is not None:
        user_data = {
            "name": first_name + " " + last_name,
            "mobile_number": mobile_number,
            "email": email,
            "company_name": company_name,
            "designation": designation,
            "password": password,
        }
        result = insert_user(db, user_data)
        return result
    else:
        return "Unable to connect to database."

def login(email, password):
    db = create_connection()
    if db is not None:
        user_id, user = authenticate_user(db, email, password)
        if user:
            return True, user_id
        else:
            return False, None
    else:
        return False, None


def validate_mobile_number(mobile_number):
    pattern = r"^\d{10}$"
    if re.match(pattern, mobile_number):
        return True
    else:
        return False

def validate_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if re.match(pattern, email):
        return True
    else:
        return False

def validate_password(password):
    if len(password) >= 8:
        return True
    else:
        return False

def insert_chat_message(db, user_id, session_id, question, answer, feedback=""):
    collection = db[CHAT_HISTORY_COLLECTION]

    # Check if a chat document already exists for the user
    existing_chat = collection.find_one({"user_id": user_id, "session_id": session_id})

    if existing_chat:
        # If a chat document exists, update it with the new conversation
        new_conversation = {
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "message_length": {
                "question_length": len(question.split()),
                "answer_length": len(answer.split())
            },
            # "user_engagement": get_user_engagement_info(session_id, user_id)


        }
        try:
            collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {"$push": {"conversation": new_conversation}}
            )
            return True
        except Exception as e:
            return str(e)
    else:
        # If no chat document exists, create a new one
        chat_data = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "session_id": session_id,   
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "conversation": [
                {
                    "question": question,
                    "answer": answer,
                    "feedback": feedback,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "message_length": {
                        "question_length": len(question.split()),
                        "answer_length": len(answer.split())
                    },
                    # "user_engagement": get_user_engagement_info(session_id, user_id)

                }
            ]
        }
        try:
            collection.insert_one(chat_data)
            return True
        except Exception as e:
            return str(e)


def update_chat_history(db, user_id, question, answer):
    collection = db[CHAT_HISTORY_COLLECTION]
    try:
        collection.update_one(
            {"user_id": user_id},
            {"$push":{"conversation": {"question": question,
                                       "answer": answer,
                                       "timestamp": datetime.datetime.now(datetime.timezone.utc),
                                       "feedback": None
                                       }
                    }                              
            },
            upsert=True
        )
        return True
    except Exception as e:
        return str(e)

def get_chat_history(db, user_id):
    collection = db[CHAT_HISTORY_COLLECTION]
    try:
        chat_history = collection.find_one({"user_id": user_id})
        if chat_history:
            return chat_history["conversation"]
        else:
            return []
    except Exception as e:
        return str(e)
    

def get_user_name(user_id):
    client = MongoClient(DATABASE_URI)
    db = client[DATABASE_NAME]
    collection = db["users"] # Replace "users" with the name of your user collection
    user = collection.find_one({"_id": user_id})
    if user:
        return user.get("name", "Unknown")  # Assuming the user document has a "name" field
    else:
        return "Unknown"



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
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
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
    with st.form("feedback_form"):
        user_feedback = st.text_input("Please provide your feedback: ")
        submitted_input = st.form_submit_button(
            "Submit question", on_click=create_feedback
        )


# Setting up logging configuration
# logging.basicConfig(level=logging.INFO)


def user_input(user_question, chat_history):
    # def user_input(user_question):
    corrected_chat_history = []
    for msg in chat_history:
        if "human" in msg:
            corrected_chat_history.append(("human", msg["human"]))
        elif "ai" in msg:
            corrected_chat_history.append(("ai", msg["ai"]))
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
        max_tokens=256,
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

    or if asked for digital input generate following as json
di_config:
  publisher:
    destination: null
    sampling_rate: 60
    debug: 0
  DiChannel:
    item:
      pin_no: 1
      pin_name: device name
      peripheral_id: '123456789123456789'
      uuid: 8e37d75f-faf5-11ee-88ce-60b6e10ad793


    generate json for MODBUS RTU device configuration if requested:
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

      generate below json for MODBUS RTU device configuration with register mapping if asked:
export_device:
  device_config:
    - devicename: MFM376
      protocol: modbusrtu
      address: ""
  configlist:
    MFM376:
      publisher:
        destination: []
        protocol: modbus-serial
        debug: 0
        sample_rate: "60"
        devicename: null
      tcpip_port: null
      serial_port:
        device: "/dev/ttyO1"
        baud_rate: 9600
        parity: N
        data_bits: 8
        stop_bits: 1
        port_timeout: 3000
      enable: 1
      query:
        - slave_id: 1
          start_address: 0
          input_type: 4
          num_of_registers: "16"
          peripheral_name: Voltage
          interval: 1000
          peripheral_id: "1234567891234567899"
          data_share: Query
          querycnt: 0
          parameter:
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Voltage V1N
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "190aec7f-11b8-11ef-96d0-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Voltage V2N
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "190af09b-11b8-11ef-96d0-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Voltage V3N
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "190af18d-11b8-11ef-96d0-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Average Voltage LN
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "342fa41a-11ba-11ef-a210-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Voltage V12
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "342fa5bb-11ba-11ef-a210-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Voltage V23
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "342fa6a8-11ba-11ef-a210-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Voltage V31
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "342fa77d-11ba-11ef-a210-60b6e10ad793"
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Average Voltage LL
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "342fa84f-11ba-11ef-a210-60b6e10ad793"
          isAddQuery: false
          totQuery: 0
          queryinterval: "100"
        - slave_id: 1
          start_address: 50
          input_type: 4
          num_of_registers: "2"
          peripheral_name: Frequency
          interval: 1000
          peripheral_id: "1234567891234567899"
          data_share: Query
          querycnt: 0
          parameter:
            - data_type: f
              adjustment_factor: 1
              converter: d
              property_name: Frequency
              param: 0
              isAddParam: false
              totalParam: 0
              uuid: "d9a5a593-11ba-11ef-a000-60b6e10ad793"
          isAddQuery: false
          totQuery: 0
          queryinterval: "100"
      devicename: MFM376
      

    generate below json for Modbus TCP/IP device configuration if asked:
export_device:
  device_config:
    - devicename: hjjlkj
      protocol: modbustcpip
      address: ""
  configlist:
    hjjlkj:
      version: "1.0.0"
      publisher:
        destination: []
        protocol: modbus-tcpip
        sample_rate: 60
        debug: 0
      devicename: meter1
      enable: 0
      tcpip_port:
        ip: "199.199.50.219"
        port: 502
        port_timeout: 3000
      query: []
   






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
        groq_chat,
        doc_db.as_retriever(search_kwargs={"k": 7}),
        contextualize_q_prompt,
    )
    # logging.info("Created history-aware retriever.")

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.2,
        max_output_tokens=2048,
        verbose=True,
        google_api_key=google_api_key,
        convert_system_message_to_human=True,
        # safety_settings={
        #                           HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_DANGEROUS:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_DEROGATORY:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_HARASSMENT:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_HATE_SPEECH:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_MEDICAL:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_SEXUAL:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_TOXICITY:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_UNSPECIFIED:HarmBlockThreshold.BLOCK_NONE,
        #                           HarmCategory.HARM_CATEGORY_VIOLENCE:HarmBlockThreshold.BLOCK_NONE
        # }
    )

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

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

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

    response = conversational_rag_chain.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": id}},
    )

    with open("user_question_response.txt", "a") as txt_file:
        txt_file.write(f"User Question: {user_question}\n")
        txt_file.write(f"Response: {response['answer']}\n")
        txt_file.write("\n")
    # logging.info("Saved response to text file.")

    return response["answer"]


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
        downloaded_files.append(download_pdf(link, directory=directory))
    return downloaded_files


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
        pdf_name = st.session_state.pdf_name  # Get pdf_name from session state
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
    if col2.button("No", key="search_button", on_click=search_datasheets):
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


def create_database():
    os.makedirs("database/user", exist_ok=True)
    conn = sqlite3.connect("database/user/user.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, response TEXT, feedback TEXT)"""
    )
    conn.commit()
    conn.close()


# Function to insert data into the database
def insert_feedback(question, response, feedback):
    conn = sqlite3.connect("database/user/user.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (question, response, feedback) VALUES (?, ?, ?)",
        (question, response, feedback),
    )
    conn.commit()
    conn.close()


def show_cimcopilot_page(user_id, session_id):
    st.header("CIMCOPILOT")
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
    
    db = create_connection()

    # memory = ChatMessageHistory()
    memory = ConversationBufferMemory()

    # Folder path where PDF and DOC files are stored
    folder_path = "./CIMdata"

    # Fetch PDF and DOC files from the folder
    pdf_files, doc_files = fetch_files_from_folder(folder_path)

    # Function to extract text from PDF files
    pdf_texts = []
    for pdf_file in pdf_files:
        pdf_texts.append(get_pdf_text([pdf_file]))

    # Get text chunks
    text_chunks = get_text_chunks(" ".join(pdf_texts))

    get_vector_store(text_chunks)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {"input": message["human"]}, {"output": message["ai"]}
            )

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # Display chat messages from history on app rerun
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # new_chat_id = f'{time.time()}'
    # MODEL_ROLE = "ai"
    # AI_AVATAR_ICON = "✨"

    # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # new_chat_id = f"{current_time}"

    # try:
    #     past_chats: dict = joblib.load("data/past_chats_list")
    # except:
    #     past_chats = {}
    # if "chat_history" not in st.session_state:
    #     st.session_state["chat_history"] = {}

    # def create_new_chat(chat_id):
    #     st.session_state["chat_history"][chat_id] = []

    # def add_message_to_chat(chat_id, message, role):
    #     if chat_id not in st.session_state["chat_history"]:
    #         create_new_chat(chat_id)
    #     st.session_state["chat_history"][chat_id].append(
    #         {"role": role, "content": message}
    #     )

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history"
        )

    # if not st.session_state.get("logged_in"):
    #     st.error("Please log in to access this page.")
    #     st.stop()

    with st.sidebar:
        image = "./images/CIMcopilot Logo-01.png"
        st.image(image, use_column_width=True)
        toggle_state = st.toggle("Find datasheet online")
        st.title("Upload datasheet")

        # if st.button("Logout"):
        #     st.session_state.logged_in = False
        #     st.experimental_rerun()
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
        # st.write("# Past Chats")
        # if st.session_state.get("chat_id") is None:
        #     st.session_state.chat_id = st.selectbox(
        #         label="Pick a past chat",
        #         options=[new_chat_id] + list(past_chats.keys()),
        #         format_func=lambda x: past_chats.get(x, "New Chat"),
        #         placeholder="_",
        #     )
        # else:
        #     st.session_state.chat_id = st.selectbox(
        #         label="Pick a past chat",
        #         options=[new_chat_id, st.session_state.chat_id]
        #         + list(past_chats.keys()),
        #         index=1,
        #         format_func=lambda x: past_chats.get(
        #             x,
        #             (
        #                 "New Chat"
        #                 if x != st.session_state.chat_id
        #                 else st.session_state.chat_title
        #             ),
        #         ),
        #         placeholder="_",
        #     )
        # st.session_state.chat_title = f"CIMCopilot-{st.session_state.chat_id}"

    # """##################################Add Voice InPUT#####################################"""
    # voice_text = ''
    # if st.button('Record'):
    #     with st.spinner(f'Recording..'):
    #         voice_text = get_voice_input()
    #         print(voice_text)
    #     st.success("Recording completed")
    # st.text_input("Text Input", value=voice_text)
    ###################################################################################

    user_question = st.chat_input(
        "Message CIMCopilot", max_chars=10000, disabled=False
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
                st.write_stream(
                    response_generator(
                        "I found below datasheet online on your request"
                        + response
                    )
                )
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

        else:
            response = user_input(user_question, st.session_state.chat_history)
            message = {"human": user_question, "ai": response}
            # print("Before appending:", st.session_state.chat_history)
            if not isinstance(st.session_state.chat_history, list):
                st.session_state.chat_history = []
            # else:
                # print(
                #     "chat_history is not a list:",
                #     st.session_state.chat_history,
                # )
            st.session_state.chat_history.append(message)

            st.session_state.messages.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            with st.chat_message("assistant"):
                # st.write(response_generator(response))
                st.write(response)
                # st.markdown(response)
            # with st.expander(label = "chat_history", expanded=False):
            #     st.write(st.session_state.chat_history)
            #     st.download_button("chat download", json.dumps(st.session_state.chat_history, indent=4), file_name="history.json", mime="application/json")

            
         

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

    
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
            )
            if feedback:
                st.toast("Feedback saved successfully.")
            else:
                st.toast("No feedback received.")
     
            insert_chat_message(db, user_id, session_id, user_question, response, feedback)




            

    with st.sidebar:
        with st.expander(label="chat_history", expanded=False):
            # st.write(st.session_state.chat_history)
            st.download_button(
                "chat download",
                json.dumps(st.session_state.chat_history, indent=4),
                file_name="history.json",
                mime="application/json",
            )


def LoggedIn_Clicked(email, password):
    success, user_id = login(email, password)
    if success:
        user_name = get_user_name(user_id)
        st.session_state["loggedIn"] = True
        st.session_state["user_id"] = user_id
        st.session_state["session_id"] = str(uuid.uuid4())
        user_name = get_user_name(user_id)
        st.session_state["user_name"] = user_name
        st.toast("Login successful. Redirecting to CIMCOPILOT")
    else:
        st.session_state["loggedIn"] = False
        st.error("Invalid user name or password")


def show_login_page():

    if st.session_state["loggedIn"] == False:
        email = st.text_input(
            label="", value="", placeholder="Enter your user name"
        )
        password = st.text_input(
            label="", value="", placeholder="Enter password", type="password"
        )
        st.button("Login", on_click=LoggedIn_Clicked, args=(email, password))


def LoggedOut_Clicked():
    st.session_state["loggedIn"] = False
    st.session_state["user_id"] = None
    st.session_state["session_id"] = None


def show_logout_page():
    # loginSection.empty();
    # with logOutSection:
    st.button("Log Out", key="logout", on_click=LoggedOut_Clicked)


def show_signup_page():
    
    st.header("Sign Up")
    first_name = st.text_input("First Name")
    last_name = st.text_input("last Name")
    # name = first_name + last_name
    mobile_number = st.text_input("Mobile Number")
    email = st.text_input("Email")
    company_name = st.text_input("Company Name")
    designation = st.text_input("Designation")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign Up"):
        if not first_name:
            st.error("Please enter your name.")
        if not last_name:
            st.error("Please enter your last name") 
        elif not validate_mobile_number(mobile_number):
            st.error("Please enter a valid 10-digit mobile number.")
        elif not validate_email(email):
            st.error("Please enter a valid email address.")
        elif not company_name:
            st.error("Please enter your company name.")
        elif not designation:
            st.error("Please enter your designation.")
        elif not validate_password(password):
            st.error("Password must be at least 8 characters long.")
        else:
            result = signup(
                first_name, last_name, mobile_number, email, company_name, designation, password
            )
            if result == True:
                st.session_state["signup_complete"] = True
                st.success("Sign Up completed")
            else:
                st.error(f"Error: {result}")



def main():
    # st.set_page_config(page_title="CIMCOPILOT")

    if "loggedIn" not in st.session_state:
        st.session_state["loggedIn"] = False
    if "signup_complete" not in st.session_state:
        st.session_state["signup_complete"] = False
    # if "user_name" not in st.session_state:
    #     st.session_state["user_name"] = ""
    if "login_time" not in st.session_state:
        st.session_state["login_time"] = None
    if "logout_time" not in st.session_state:
        st.session_state["logout_time"] = None

    if st.session_state["loggedIn"]:
        with st.sidebar:
            st.header(f"Welcome, :blue[{st.session_state['user_name']}]!")
            show_logout_page()
        show_cimcopilot_page(st.session_state["user_id"], st.session_state["session_id"])
    else:
        if st.session_state["signup_complete"]:
            # Automatically redirect to the login page
            st.session_state["signup_complete"] = (
                False  # Reset to avoid redirect loop
            )
            show_login_page()
        else:
            with st.sidebar:
                image = "./images/CIMcopilot Logo-01.png"
                st.image(image, use_column_width=True)
            page = st.sidebar.radio("Navigation", ["Sign Up", "Login"])
            if page == "Sign Up":
                show_signup_page()
            elif page == "Login":
                show_login_page()


if __name__ == "__main__":
    main()
