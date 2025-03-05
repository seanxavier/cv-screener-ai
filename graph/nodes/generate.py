from typing import Any, Dict

from graph.chains.generation import generation_prompt, generation_prompt_runnable
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("--- GENERATE ---")
    job_posting = state["job_posting"]
    resumes = state["resumes"]
    question = state["question"]
    individual_assessments = state["individual_assessments"]
    
    
    generation = generation_prompt_runnable.invoke({"job_posting": job_posting, "resumes":resumes,"individual_assessments":individual_assessments, "question" : question })
    print(f"generation: {generation}")
    return {"job_posting": job_posting, "resumes":resumes, "question" : question, "generation":generation}
    