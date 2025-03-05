from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


from graph.consts import GENERATE
from graph.nodes import generate
from graph.state import GraphState

workflow = StateGraph(GraphState)

workflow.add_node(GENERATE, generate)
workflow.set_entry_point(GENERATE)
workflow.add_edge(GENERATE, END)

# responsible for persisting the memory of state upon each graph execution
# only stores in in-memory, meaning it will be gone after each session
memory = MemorySaver()

# graphApp = workflow.compile(checkpointer=memory)
graphApp = workflow.compile()

graphApp.get_graph().draw_mermaid_png(output_file_path="graph.png")
