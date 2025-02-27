import streamlit as st
import logging
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

from dotenv import load_dotenv

load_dotenv()

IBM_CLOUD_URL= os.environ["IBM_CLOUD_URL"]
IBM_CLOUD_API_KEY=os.environ["IBM_CLOUD_API_KEY"]
WATSONX_PROJECT_ID=os.environ["WATSONX_PROJECT_ID"]

# supported_models= ['codellama/codellama-34b-instruct-hf', 
#                    'google/flan-t5-xl', 'google/flan-t5-xxl', 
#                    'google/flan-ul2', 
#                    'ibm/granite-13b-instruct-v2', 
#                    'ibm/granite-20b-code-instruct', 
#                    'ibm/granite-20b-multilingual', 
#                    'ibm/granite-3-2-8b-instruct', 
#                    'ibm/granite-3-2-8b-instruct-preview-rc', 
#                    'ibm/granite-3-2b-instruct', 
#                    'ibm/granite-3-8b-instruct', 
#                    'ibm/granite-34b-code-instruct', 
#                    'ibm/granite-3b-code-instruct', 
#                    'ibm/granite-8b-code-instruct', 
#                    'ibm/granite-guardian-3-2b', 
#                    'ibm/granite-guardian-3-8b', 
#                    'ibm/granite-vision-3-2-2b', 
#                    'meta-llama/llama-2-13b-chat', 
#                    'meta-llama/llama-3-1-70b-instruct', 
#                    'meta-llama/llama-3-1-8b-instruct', 
#                    'meta-llama/llama-3-2-11b-vision-instruct', 
#                    'meta-llama/llama-3-2-1b-instruct', 
#                    'meta-llama/llama-3-2-3b-instruct', 
#                    'meta-llama/llama-3-2-90b-vision-instruct', 
#                    'meta-llama/llama-3-3-70b-instruct', 
#                    'meta-llama/llama-3-405b-instruct', 
#                    'meta-llama/llama-guard-3-11b-vision', 
#                    'mistralai/mistral-large', 
#                    'mistralai/mixtral-8x7b-instruct-v01']

generate_params = {
    GenParams.MAX_NEW_TOKENS: 100
}

model_inference = ModelInference(
    model_id="ibm/granite-3-2-8b-instruct",
    params=generate_params,
    credentials=Credentials(
        api_key = IBM_CLOUD_API_KEY,
        url = IBM_CLOUD_URL),
        project_id=WATSONX_PROJECT_ID
    )











##### -----  START STREAMLIT APP  ######


# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="CV Screener",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Candidate Assessment powered by watsonx 💬")

# Sidebar contents
with st.sidebar:
    st.title("Candidate Assessment App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
 
    ''')
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')

    
uploaded_cv_files = st.file_uploader("Choose Job Posting PDF files", accept_multiple_files=True)
uploaded_cv_files = st.file_uploader("Choose Candidate CV PDF files", accept_multiple_files=True)


if st.text_input(label="HI"):
    st.write_stream(model_inference.generate_text_stream(prompt="what is llm?"))


