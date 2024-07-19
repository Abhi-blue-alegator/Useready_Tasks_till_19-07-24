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
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision

)

def main():

    st.set_page_config(page_title="Ask your PDF")

    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
    if "api_key" in st.session_state:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        st.success("API Key saved successfully")

    st.header("Ask your PDF")
    with st.sidebar:
        st.markdown("## Select your PDF and ask a question related to its content")

    pdf = st.file_uploader("Choose a file", type="pdf")

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

        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your PDF")

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

            questions = ["what are the common uses of databricks?",
                            "when can we integrate databricks into streamlit?"]
            
            ground_truths=[["Databricks is commonly used for big data analytics and processing, particularly for Apache Spark-based data processing tasks. It provides a unified analytics platform that integrates with various data sourcesand supports collaborative data science workflows. Additionally, it's utilized for machine learning, data engineering, and real-time analytics applications."],
                            ["We can integrate Databricks into your Streamlit app for data processing tasks by leveraging Databricks as a backend service. For example, you can use Databricks for heavy-duty data transformations, machine learning model training, or large-scale data analysis tasks."]]
            

            answers=[]
            contexts=[]

            for query in questions:
                answers.append(rag_chain.invoke(query))
                contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

            
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truths}
                


            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset= dataset,
                metrics=[faithfulness, 
                            answer_relevancy, 
                            context_recall, 
                            context_precision
                            ],
                    )
            

            df = result.to_pandas()
            st.write(df)



if __name__ == "__main__":
    main()



