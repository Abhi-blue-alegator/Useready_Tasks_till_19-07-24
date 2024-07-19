import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from pprint import pprint
from typing import List
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_utilization
)

# Load the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Streamlit UI
st.sidebar.title("LangGraph Chatbot")

urls_input = st.sidebar.text_area("Enter URLs (one per line):")
question = st.text_input("Enter your question:")

# Function to initialize retriever
def initialize_retriever(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()



urls = urls_input.strip().split("\n")
retriever = initialize_retriever(urls)

# LLM, Prompt, and Chain
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api_key)

# Chains
rag_chain1 = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
)
rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"context": retriever, "question": question})

# RAGAS Evaluation
questions = [question]
answers, contexts = [], []

for query in questions:
    answers.append(rag_chain1.invoke(query))
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

data = {"question": questions, "answer": answers, "contexts": contexts}
dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_relevancy, context_utilization])
df = result.to_pandas()
# Retrieval Grader
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
    num_score: float = Field(description="Score from 0 to 1")

system = """You are a grader assessing relevance of a retrieved document to a user question. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")])
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)

docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
ret_score = retrieval_grader.invoke({"question": question, "document": doc_txt})
ret_output = {
    "binary_score": ret_score.binary_score,
    "num_score": result['context_relevancy']
}

# Hallucination Grader
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
    num_score: float = Field(description="Score from 0 to 1")

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.""" 
hallucination_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")])
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)
hal_score = hallucination_grader.invoke({"documents": docs, "generation": generation})
hal_output = {
    "binary_score": hal_score.binary_score,
    "num_score": result['faithfulness']
}

# Answer Grader
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
    num_score: float = Field(description="Score from 0 to 1")

system = """You are a grader assessing whether an answer addresses / resolves a question 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")])
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)
ans_score =answer_grader.invoke({"question": question, "generation": generation})
ans_output = {
    "binary_score": ans_score.binary_score,
    "num_score": result['answer_relevancy']
}

# Question Rewriter
system = """You a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")])
question_rewriter = re_write_prompt | llm | StrOutputParser()


# Define graph state and functions
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        num_score = ret_output["num_score"]
        if grade == "yes" or num_score > 0.5:
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state):
    filtered_documents = state["documents"]
    if not filtered_documents:
        return "transform_query"
    else:
        return "generate"

def grade_generation_v_documents_and_question(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    num_score = hal_output['num_score']
    if grade == "yes" or num_score > 0.5:
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        num_score = ans_output['num_score']
        if grade == "yes" or num_score > 0.5:
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {"not supported": "generate", "useful": END, "not useful": "transform_query"})
app = workflow.compile()




if st.button("Submit"):
    if urls_input and question:
        urls = urls_input.strip().split("\n")
        retriever = initialize_retriever(urls)
        inputs = {"question": question}
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")
                pprint(value)
        st.write(result)
        st.write(df)        
        st.write(f"Answer: {value['generation']}")

    
        df1 = df.iloc[:, 3:]
        # Plotting visualizations
        st.header("Visualizations")

        # Bar chart
        st.subheader("Bar Chart")
        st.bar_chart(df1)


    else:
        st.write("Please enter both URLs and a question.")
