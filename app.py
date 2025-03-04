import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import logging
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader

import tempfile

import pandas as pd
import json

from dotenv import load_dotenv
import time
load_dotenv()

IBM_CLOUD_URL= os.environ["IBM_CLOUD_URL"]
IBM_CLOUD_API_KEY=os.environ["IBM_CLOUD_API_KEY"]
WATSONX_PROJECT_ID=os.environ["WATSONX_PROJECT_ID"]

# Most GENAI logs are at Debug level.
#  "DEBUG", "INFO", "WARNING"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

DEFAULT_MODEL='ibm/granite-3-2-8b-instruct'
SUPPORTED_MODELS= [
                    # 'codellama/codellama-34b-instruct-hf', 
                #    'google/flan-t5-xl', 
                #    'google/flan-t5-xxl', 
                #    'google/flan-ul2', 
                #    'ibm/granite-13b-instruct-v2', 
                #    'ibm/granite-20b-code-instruct', 
                #    'ibm/granite-20b-multilingual', 
                   'ibm/granite-3-2-8b-instruct', 
                #    'ibm/granite-3-2-8b-instruct-preview-rc', 
                #    'ibm/granite-3-2b-instruct', 
                #    'ibm/granite-3-8b-instruct', 
                #    'ibm/granite-34b-code-instruct', 
                #    'ibm/granite-3b-code-instruct', 
                #    'ibm/granite-8b-code-instruct', 
                #    'ibm/granite-guardian-3-2b', 
                #    'ibm/granite-guardian-3-8b', 
                #    'ibm/granite-vision-3-2-2b', 
                #    'meta-llama/llama-2-13b-chat', 
                #    'meta-llama/llama-3-1-70b-instruct', 
                #    'meta-llama/llama-3-1-8b-instruct', 
                #    'meta-llama/llama-3-2-11b-vision-instruct', 
                #    'meta-llama/llama-3-2-1b-instruct', 
                #    'meta-llama/llama-3-2-3b-instruct', 
                #    'meta-llama/llama-3-2-90b-vision-instruct', 
                #    'meta-llama/llama-3-3-70b-instruct', 
                #    'meta-llama/llama-3-405b-instruct', 
                #    'meta-llama/llama-guard-3-11b-vision', 
                   'mistralai/mistral-large', 
                #    'mistralai/mixtral-8x7b-instruct-v01'
                   ]




def getLLM(model_id="meta-llama/llama-3-3-70b-instruct", max_new_tokens=2000, min_new_tokens=20, decoding_method="greedy", stop_sequence=["<|endoftext|>","</s>"]) -> ModelInference:
    """
    Initializes LLMs, this enables choosing LLM in streamlit
    
    Default values:
    model_id= 'meta-llama/llama-3-3-70b-instruct'
    max_new_tokens=2000
    min_new_tokens=20
    decoding_method="greedy"
    stop_sequence=[]
    """
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.MIN_NEW_TOKENS:min_new_tokens,
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.STOP_SEQUENCES: stop_sequence
    }
    
    return ModelInference(
        model_id=model_id,
        params=generate_params,
        credentials=Credentials(
            api_key = IBM_CLOUD_API_KEY,
            url = IBM_CLOUD_URL
            ),
            project_id=WATSONX_PROJECT_ID
        )

# DEPRECATED
generate_text_prompt = PromptTemplate.from_template(
    """
    You are an experience HR Recruiter. Respond in a markdown format. Your task is to assess candidates based on their resume and the job posting.
    Your assessment must include 3 main points: 
        1. Suitability: High, Medium, Low
        2. Score: 1-100
        3. Recommended: Yes/No
        
    For example:
    Name | Suitability | Score | Recommended
    
    Include a detailed assessment report on how you came up with the 3 main points.
    
    
    Job Posting:
    {job_posting_text}
    
    Candidate Resume:
    {candidate_resume_text}
    """
    )   

json_schema= """
```json
    {
    "type": "object",
    "properties": {
        "name": {
        "type": "string"
        },
        "suitability": {
        "type": "string",
        "enum": ["High", "Medium", "Low"]
        },
        "score": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100
        },
        "recommended": {
        "type": "string",
        "enum": ["Yes", "No"]
        },
        "detailed_assessment": {
        "type": "string"
        }
    },
    "required": ["suitability", "score", "recommended", "detailed_assessment"]
    }
    
"""    
generate_json_prompt=PromptTemplate.from_template(
    """
    You are an experience HR Recruiter. Your task is to assess candidates based on their resume and the job posting.
    Your assessment must include 3 main points: 
        1. Name: Name of the candidate
        2. Suitability: High, Medium, Low
        3. Score: 1-100
        4. Recommended: Yes/No
        5. Detailed assessment: a detailed assessment report on how you came up with the 3 main points. Discuss a detailed report on suitability, score, and recommendation. Detailed assessment must be in a markdown format but it will be stored in a single-line string.   
   
    Your response must be ONE valid JSON only in this JSON specification:
    {json_schema}
    
    Do not return codes and code blocks.
    ONLY RETURN THE JSON STRING, DO NOT RETURN ANYTHING ELSE.
    
    # Job Posting:
    {job_posting_text}
    
    # Candidate Resume:
    {candidate_resume_text}
    
    # JSON response:
    """
    )   


def extract_text_from_pdfs(uploaded_files):
    """Extracts text from multiple PDF files and stores it in a dictionary."""

    all_extracted_text = {}  # Store extracted text for each file

    # IF single file only, st.file_uploader returns UploadedFile obj if it's restricted to accept single file
    # ELSE, if multiple files, it returns a list of UploadedFiles
    if not (isinstance(uploaded_files, UploadedFile)):
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                print(f"uploaded_file: {uploaded_file}")
                
                with st.spinner(f"Reading resumes ({index+1}/{len(uploaded_files)}): {uploaded_file.name}...", show_time=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name

                    loader = PDFPlumberLoader(temp_file_path)
                    documents = loader.load()
                    extracted_text = "\n".join([doc.page_content for doc in documents])

                    all_extracted_text[uploaded_file.name] = extracted_text

            except Exception as e:
                st.error(f"An error occurred with {uploaded_file.name}: {e}")
            finally:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    else:
        try:
            print(f"uploaded_file: {uploaded_files}")
            with st.spinner(f"Reading job posting: {uploaded_files.name}... ", show_time=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_files.read())
                    temp_file_path = temp_file.name

                loader = PDFPlumberLoader(temp_file_path)
                documents = loader.load()
                extracted_text = "\n".join([doc.page_content for doc in documents])

                all_extracted_text[uploaded_files.name] = extracted_text

        except Exception as e:
            st.error(f"An error occurred with {uploaded_files.name}: {e}")
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        

    return all_extracted_text



##### -----  START STREAMLIT APP  ######


def streamlit_app():
    st.set_page_config(
        page_title="CV Screener",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.header("Candidate Assessment powered by watsonx 💬")

    # Sidebar contents
    with st.sidebar:
        st.title("Candidate Assessment App")
        st.markdown('''
        ## About
        This app is an LLM-powered candidate assessment app built using:
        - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
        - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
    
        ''')
        st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
        st.title("Model Inference Parameters")

        model_id = st.selectbox("Select Model ID", SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(DEFAULT_MODEL))
        max_new_tokens = st.number_input("Max New Tokens", min_value=1, value=1000, step=1)
        min_new_tokens = st.number_input("Min New Tokens", min_value=0, value=20, step=1)
        decoding_method = st.selectbox("Decoding Method", ["greedy", "sampling"])
        # stop_sequence_str = st.text_input("Stop Sequence (optional)")
        # if stop_sequence_str:
        #     stop_sequence = [s.strip() for s in stop_sequence_str.split(',')]
        # else:
        #     stop_sequence = [""]  
        
    # we initialize llm from here because we'll get the details from the input fields in sidebar. 
    llm = getLLM(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        decoding_method=decoding_method,
        # stop_sequence=stop_sequence
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Upload job posting and cadidate CVs then click generate.")    
        uploaded_job_file = st.file_uploader("Choose Job Posting PDF files",  type=["pdf"], accept_multiple_files=False)
        uploaded_cv_files = st.file_uploader("Choose Candidate CV PDF files",  type=["pdf"], accept_multiple_files=True)

        if uploaded_job_file is not None and uploaded_cv_files is not None:
            generate_button_disabled = False
        else:
            generate_button_disabled = True
            
            
        assessment_report = []
        if st.button("Generate Assessment", disabled=generate_button_disabled):
            
            # Exctract the pdf files
            extracted_job_file_data = extract_text_from_pdfs(uploaded_job_file)
            job_posting_extracted_text = next(iter(extracted_job_file_data.values()))
            extracted_cv_file_data = extract_text_from_pdfs(uploaded_cv_files)
            
            # Start generating the assessments
            st.subheader("Detailed Assessments", divider="gray")
            for index, (key, value) in enumerate(extracted_cv_file_data.items()):
                with st.spinner(f"Analyzing resume ({index+1}/{len(extracted_cv_file_data)}): {key}...", show_time=True):
                    json_string_output_raw=llm.generate_text(
                        prompt=generate_json_prompt.format(json_schema=json_schema,candidate_resume_text=value, job_posting_text=job_posting_extracted_text),
                        )
                    logger.debug(f"json_string_output_raw:")
                    logger.debug(json_string_output_raw)
                    json_string_output_clean = json_string_output_raw[9:-3]
                    logger.debug(f"json_string_output_clean:")
                    logger.debug(json_string_output_clean)
                    jsonify_output = json.loads(json_string_output_clean)
                    logger.debug(f"jsonify_output:")
                    logger.debug(jsonify_output)
                    assessment_report.append(jsonify_output)
                    
                    # st.write(jsonify_output)
                    
                    with st.expander(f"{jsonify_output['name']}"):
                        st.write(f"Candidate Name: {jsonify_output['name']}")
                        st.write(f"Suitability: {jsonify_output['suitability']}")
                        st.write(f"Score: {jsonify_output['score']}")
                        st.write(f"Recommended: {jsonify_output['recommended']}")
                        
                        st.write("Detailed Assessment")
                        st.write(jsonify_output["detailed_assessment"])

            st.subheader("Assessment Summary", divider="gray")
            df = pd.DataFrame(assessment_report)
            # print(df)
            # filter out the detailed assessment in the table summary.
            df_overview = df[["name", "suitability", "score", "recommended"]]
            st.dataframe(df_overview, column_config={
                "Name": st.column_config.Column(label="Candidate Name",width="medium"), 
                "Suitability": st.column_config.Column(label="Suitability",width="small"),
                "Score": st.column_config.Column(label="Score",width="small"),
                "Recommended": st.column_config.Column(label="Recommended",width="small"),
                # "Detailed Assessment": st.column_config.Column(label="Detailed Assessment",width="large"),
            })
    
    with col2:
        with st.container():
            st.subheader("Assistant")
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Create a container for the chat history with scrollable behavior
            chat_history_container = st.container( key="chat-history" )

            # Display chat messages from history on app rerun
            with st.container():
                with chat_history_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

            # React to user input
            with st.container(key="stChatInputContainer"):
                if prompt := st.chat_input("What can I help you with?", key="user-input"):
                    # Display user message in chat message container
                    with chat_history_container:
                        st.chat_message("user").markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    response = f"Echo: {prompt}"
                    # Display assistant response in chat message container
                    with chat_history_container:
                        time.sleep(1)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Add CSS to ensure chat input stays at the bottom and chat history scrolls
            st.markdown(
                """
                <style>
                .st-key-user-input {
                    position: fixed;
                    bottom: 10px;
                    width: 100%;
                }
                .st-key-parent-chat-history > div {
                    width: 500px;
                }
                .st-key-chat-history {
                    position: fixed;
                    bottom: 110;
                    width: auto;
                    height: 500px;
                    overflow: auto;
                    border: 1px solid white
                    border-radius: 10px;
                    
                    
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        
if __name__ == "__main__":
    streamlit_app()


