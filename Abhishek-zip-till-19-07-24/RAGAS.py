import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from dotenv import load_dotenv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    context_utilization,
    context_relevancy

)
text = ""
def main():

    st.set_page_config(page_title="Evaluation With RAGAS", page_icon=":book:")
    # Load the .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.session_state.api_key = api_key
    if "api_key" in st.session_state:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key


# Initialize the session state for questions if it doesn't exist
    if 'questions' not in st.session_state:
        st.session_state.questions = []

    # Function to add a new question from the input field
    def add_question():
        new_question = st.session_state.new_question
        if new_question:
            st.session_state.questions.append(new_question)
            st.session_state.new_question = ""  # Clear the input field

    # Function to remove a question
    def remove_question(index):
        st.session_state.questions.pop(index)

    # Display a header
    st.header("Dynamic Question Input")

    # Input field for the new question
    st.text_input("Enter a new question:", key='new_question')

    # Add button to add the new question
    st.button("Add Question", on_click=add_question)

    # Display the current questions with input fields
    for i, question in enumerate(st.session_state.questions):
        col1, col2 = st.columns([0.9, 0.1])
        st.session_state.questions[i] = col1.text_input(f"Question {i+1}", question, key=f"question_{i}")
        if col2.button("Remove", key=f"remove_{i}"):
            remove_question(i)

    # Optionally, you can display the questions variable as well
    st.write("Questions variable:", st.session_state.questions)


    st.sidebar.header("Evaluation With RAGAS")
    

    pdf = st.sidebar.file_uploader("Choose a file", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

    
    if "api_key" in st.session_state:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        try:
            chunks = text_splitter.split_text(text)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)

        except:
            st.sidebar.info("Please provide the PDF")

        

        user_question = st.sidebar.button("submit")

        if user_question:
            retriever = vectorstore.as_retriever()

            llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)

            template = """ you are a helpful pdf assistant. 
                    Given the following pdf, answer the question based on the context.
                    If you don't know the answer, just say that you don't know. 
                    Do not make up an answer.

                    Question: {question}
                    Context: {context}
                    
                    Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                |StrOutputParser()
                
            )


            
            
            
            answers=[]
            contexts=[]

            questions = st.session_state.questions

            for query in questions:
                answers.append(rag_chain.invoke(query))
                contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

            
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts}
                


            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset= dataset,
                metrics=[faithfulness, 
                            answer_relevancy,  
                            context_relevancy,
                            context_utilization
                            ],
                    )
            

            df = result.to_pandas()
            st.write(df)

            df1 = df.iloc[:, 3:]
            # Plotting visualizations
            st.header("Visualizations")

            # Bar chart
            st.subheader("Bar Chart")
            st.bar_chart(df1)

            # Line chart
            st.subheader("Line Chart")
            st.line_chart(df1)

                        # Heatmap
            st.subheader("Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df1, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)





if __name__ == "__main__":
    main()



