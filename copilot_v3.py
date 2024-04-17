# Standard library imports
import os
import json
import pandas as pd
import requests
from streamlit_text_rating.st_text_rater import st_text_rater


# Third-party imports
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain 
from langchain.chains import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader,Docx2txtLoader,UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.utilities import GoogleSerperAPIWrapper

# os.environ['GOOGLE_API_KEY'] = "AIzaSyDL5KKAzI2z7ue7g_s62i4ZgfDw70gnc9A"
os.environ['GOOGLE_API_KEY'] = "AIzaSyDYmei4WbkEX9p_6rF_RaANkl72DgxEIBQ"

genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
# data_path = "/home/cimcon/Documents/Rajveer Rathod/CIMCopilot/chatwith datasheet/CIMdata"

loaders = {
    '.pdf': PyMuPDFLoader,
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
    '.docx':UnstructuredWordDocumentLoader, 
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""

As an adept in analyzing technical specifications, configurations for a broad array of edge devices like flowmeters, 
pressure sensors, conductivity sensors, Force Sensors, CO2 sensors, Humidity, Transmitters, 
Tilt sensors, Level Transmitters, Ph Transmitters, Position Transmitters, Optical Sensors, 
Vibrations sensors, and Speed Transmitters, your proficiency lies in extracting its operational
parameters and specifications. You possess extensive knowledge across various manufacturers,
encompassing crucial details such as make, model number, power consumption, processing capabilities,
connectivity options, input/output interfaces, operating range, and firmware/software support.

Your primary task is to offer precise and detailed information regarding edge device specifications
in response to specific queries. If the requested data is unavailable directly from the context,
you are to respond with "answer is not available in the context." Additionally,
you're equipped to provide information about the CIM10 upon request. and only if user need more information For More Information you can contact our support team on  "support@cimcondigital.com". not every response should contain this.

Regarding CIM10 Analog Input Configuration, there are two pins available,
supporting up to two analog channels at maximum. You're tasked with configuring
various values such as Range, Destination, Sampling Rate, Debug, NumOfChannels,
Enable, Pin Number, Channel Type, Name, Peripheral ID, EnggLowCal, EnggHighCal, Scalelow, and
Scalehigh. These parameters facilitate the calibration and configuration and conversion of raw and engineering values, enabling effective configuration of analog channels on the CIM10.

Regarding CIM10 Digital Input Configuration, there are two pins available, supporting upto 2 Digital Channel at maximum.
You are tasked with configuring various values such as range, Sampling Rate, Debig, Number Of Channel, Device ID. 
These parameters facilitate the calibration and configuration and conversion of raw and engineering values, enabling effective configuration of digital channels on the CIM10.

For instance, if configuring a any field device with a 4-20mA output on AI pin, the configuration settings on CIM10 would include Pin Number: 1, Sampling rate (Sec): between 1 to 86400, Destination: CIMCON Cloud, Name: user-defined, Device ID: 1122334455667788, Channel Type: current, Engg. Scale Low: 4, Engg. Scale High: 20, Scale Low: minimum value based on datasheet range, Scale High: maximum value based on datasheet range.

Below is a tabular representation of the configuration settings:

| Parameter          | Value                                   |
|-------------------|-----------------------------------------|
| Pin Number        | 1                                       |
| Sampling rate (Sec)| 1 to 86400                              |
| Destination       | CIMCON Cloud                            |
| Name              | User-defined                            |
| Device ID         | user_defined                        |
| Channel Type      | I                                |
| Engg. Scale Low   | 4                                       |
| Engg. Scale High  | 20                                      |
| Scale Low         | Minimum value based on datasheet range   |
| Scale High        | Maximum value based on datasheet range   |

or if the channel is voltage channel then


| Parameter          | Value                                   |
|-------------------|-----------------------------------------|
| Pin Number        | 1                                       |
| Sampling rate (Sec)| 1 to 86400                              |
| Destination       | CIMCON Cloud                            |
| Name              | User-defined                            |
| Device ID         | user_defined                        |
| Channel Type      | V                                 |
| Engg. Scale Low   | 0                                       |
| Engg. Scale High  | 10                                      |
| Scale Low         | Minimum value based on datasheet range   |
| Scale High        | Maximum value based on datasheet range   |

if the device is capable of connectign with Digital Input pin thenn the configuration settings on cim10 would include pin number 1, sampling rate(sec): between 1 to 86400, Destination:Any, Pin_name: user-defined

| Parameter          | Value                                   |
|-------------------|-----------------------------------------|
| Pin Number        | 1                                       |
| Sampling rate (Sec)| 1 to 86400                              |
| Destination       | CIMCON Cloud                            |
| pin_Name              | User-defined                            |
| Device ID         | user_defined                        |


Your comprehensive understanding of edge device specifications and configuration parameters makes you adept at configuring analog channels on the CIM10 effectively. If the requested data is unavailable directly from the datasheets,
you are to respond with "answer is not available in the context."
you need to generate short and to the point responses.
Context: 

{context} 

  

Question: 

{question} 
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",    temperature=0.5)
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    # chat = model.start_chat(history=[])
    # chat
     


    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # chain = create_history_aware_retriever(model, chain_type="stuff", prompt=prompt)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)
    # docs = new_db.sh
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    # st.write("Reply: ", response["output_text"])

    # if "answer is not available in the context" in response["output_text"]:
    #     # If the answer is not available, perform a Google search
    #     search_response = search_and_generate_response(user_question)
    #     st.write("Search Response: ", search_response)
    # else:
    #     st.write("Reply: ", response["output_text"])

    # json_text = response.strip('`\r\n ').removeprefix('json')
    # json_text = str(response.get('response', ''))
    # print(json.dumps(json.loads(json_text), indent=4))

    # Save user question and response in JSON format
    data = {
        "user_question": user_question,
        "response": response["output_text"]
    }
    with open("user_question_response.json", "w") as json_file:
        json.dump(data, json_file)

    # Save user question and response in text format
    with open("user_question_response.txt", "a") as txt_file:
        txt_file.write(f"User Question: {user_question}\n")
        txt_file.write(f"Response: {response['output_text']}\n")
        txt_file.write("\n")

    # Chatbot responses for greetings and small talks
    # answer = handle_greetings_and_small_talks(user_question)
    # if answer:
    #     st.write("Chatbot: ", answer)
    # else:
    #     
    return response["output_text"]

# def handle_greetings_and_small_talks(user_input):
#     greetings = ["hello", "hi", "hey", "howdy"]
#     small_talks = [
#         "how are you?",
#         "what's up?",
#         "who are you?",
#         "nice to meet you!",
#         "how can you help me?",
#         "how are you",
#         "what's up",
#         "who are you",
#         "nice to meet you",
#         "how can you help me",
#         "what can you do",
#     ]
#     responses = [
#         "I'm an AI assistant from cimcon here to help you!",
#         "I'm your virtual assistant from cimcon.",
#         "I'm a chatbot from cimcon designed to assist you with your queries from.",
#         "I'm here to provide assistance and answer your questions.",
#         "I'm an AI programmed to assist you in any way I can.",
#     ]

#     if user_input.lower() in greetings:
#         ans = random.choice(responses)
#         return "Hello! I'm **CIMCOPILOT**, your digital assistant here to guide and support you every step of the way. Whether you need help configuring field devices, need information about CIM10 and its webUI, I'm here 24/7 to assist. To get started, you can type your question."
#     elif user_input.lower() in small_talks:
#         return random.choice(responses)
#     else:
#         return None

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
    filename = os.path.join(directory, url.split('/')[-1])

    # Download PDF file
    with open(filename, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)
    return filename

def download_pdfs_from_search_results(search_query, directory="downloaded_pdfs"):
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
    pdf_links = [item['link'] for item in results.get('organic', []) if item['link'].endswith('.pdf')]

    # Download PDFs
    downloaded_files = []
    for link in pdf_links:
        downloaded_files.append(download_pdf(link, directory=directory))
    return downloaded_files


def response_to_json(response):
    # Convert response to JSON format
    json_data = {
        "request":"set_config",
        "request_parameters": response
    }
    return json_data

def save_chat_history_to_csv(chat_history):
    chat_history.to_csv("chat_history.csv", index=False)

def load_chat_history_from_csv():
    try:
        return pd.read_csv("chat_history.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["ChatID", "role", "content"])

# Function to get button label for each chat
def get_button_label(chat_df, chat_id):
    first_message = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "user")].iloc[0]["Content"]
    return f"Chat {chat_id[0:7]}: {' '.join(first_message.split()[:5])}..."

def main():
    st.set_page_config("CIMCOPILOT")
    st.header("CIMCOPILOT")
    st.text("Start your chat")

    # Folder path where PDF and DOC files are stored
    folder_path = "./CIMdata"

    # Fetch PDF and DOC files from the folder
    pdf_files, doc_files = fetch_files_from_folder(folder_path)

    if not pdf_files:
        st.warning("No PDF files uploaded. Performing internet search for datasheets...")
        # Trigger internet search and download process
        search_query = st.text_input("Enter your search query:")
        if st.button("Search"):
            pdf_files = download_pdfs_from_search_results(search_query)
    
    # Function to extract text from PDF files
    pdf_texts = []
    for pdf_file in pdf_files:
        pdf_texts.append(get_pdf_text([pdf_file]))

    # Get text chunks
    text_chunks = get_text_chunks(" ".join(pdf_texts))

    # Generate vector store
    get_vector_store(text_chunks)

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history_from_csv()

    # Display chat history buttons in sidebar
    for chat_id in st.session_state.chat_history["ChatID"].unique():
        button_label = get_button_label(st.session_state.chat_history, chat_id)
        if st.sidebar.button(button_label):
            current_chat_id = chat_id
            loaded_chat = st.session_state.chat_history[st.session_state.chat_history["ChatID"] == chat_id]
            loaded_chat_string = "\n".join(f"{row['role']}: {row['content']}" for _, row in loaded_chat.iterrows())
            st.text_area("Chat History", value=loaded_chat_string, height=300)


    user_question = st.chat_input("Message CIMCopilot", max_chars=5000)
    
    if user_question:
        # user_input(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        # st.write(f"User: {user_question}")
        with st.chat_message("user"):
            st.markdown(user_question)
        response = user_input(user_question)

        with st.chat_message("assistant"):
            # st.write(response)
            ratings = st_text_rater(text=st.write(response))
        st.session_state.messages.append({"role": "assistant", "content": response})
    save_chat_history_to_csv(st.session_state.chat_history)
        # response = json.dumps(response)

        # col1,col2,col3,col4 = st.columns([3,3,0.5,0.5])
        # with col3:
        #             if st.button(":thumbsup:"):
        #                 print("Like")
        # with col4:
        #             if st.button(":thumbsdown:"):
        #                 print("Dislike")
        # json_data = response_to_json(response)
        # st.button("Download Config")
        # if  st.button("Download Config") and isinstance(json_data, dict):
            
        #         # Save JSON to file
        #     with open("ai_config.json", "w") as json_file:
        #         json.dump(json_data, json_file)
        #         # Provide download link
        #     with open("ai_config.json", "rb") as f:
            
        #         data = f.read()
        #     b64 = base64.b64encode(data).decode()
        #     href = f'<a href="data:application/octet-stream;base64,{b64}" download="ai_config.json">Download ai_config.json</a>'
        #     st.markdown(href, unsafe_allow_html=True)

        #     # st.subheader("Generated Configuration:")
        #     # st.json(json_data)


    with st.sidebar:
        st.title("Upload datasheet")
        pdf_docs = st.file_uploader("Upload your datasheets and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                print("raw text", raw_text)
                print("###########################################################################################")
                text_chunks = get_text_chunks(raw_text)
                print("tet chunks: ", text_chunks)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()

