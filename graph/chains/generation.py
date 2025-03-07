from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSequence


import os
from dotenv import load_dotenv

load_dotenv()

WATSONX_APIKEY = os.environ["IBM_CLOUD_API_KEY"]
WATSONX_URL= os.environ["IBM_CLOUD_URL"]
WATSONX_PROJECT_ID= os.environ["WATSONX_PROJECT_ID"]

# 1st we define the llm
parameters={
    
}
llm = ChatWatsonx(
    model_id="ibm/granite-3-2-8b-instruct",
    apikey=WATSONX_APIKEY,
    url=WATSONX_URL,
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)

# 2nd we write the system prompt
system="""
You are a helpful HR recruiter. You are an expert in analysing the candidate resume or CV and the job posting.
You must answer the questions truthfully and based on the given job posting information and the candidate resume.
You must answer with details and justify your answer. You must be objective, ethical, and fair. Do not be biased and racist. 
You're only responsibility is to answer questions related to the job posting, candidate resume or CV, and related queries. 
You are not allowed to answer questions outside that scope.

Your answer must be in a Markdown format for better readibility. You can produce markdown for table format, list, and bullets. Do not return code blocks and codes.

"""

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Job posting: {job_posting} \n\n Candidate resumes: {resumes} \n\n Individual Assessments: {individual_assessments} Question: {question}")
    ]
)

# we define the hallucination_grader chain
# Sequence of Runnables, where the output of each is the input of the next.
generation_prompt_runnable: RunnableSequence = generation_prompt | llm | StrOutputParser()