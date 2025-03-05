from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        generation: LLM generation
        job_posting: the job listing information
        resumes: List of candidate's resumes
        contexts: List of contexts
        question: User query
    """
    
    question: str
    job_posting: List[dict]
    resumes: List[dict]
    individual_assessments: List[dict]
    contexts: List[str]
    generation: str


