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
from dotenv import load_dotenv



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
    if 'question' not in st.session_state:
        st.session_state.question = []

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
    st.header("Enter Your Question")

    # Input field for the new question
    st.text_input("Enter a new question:", key='new_question')


    st.sidebar.header("Healthcare Research Assistant")
    

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

            template = """ you are a helpful healthcare pdf assistant. 
                    Given the following pdf, answer the question based on the context.
                    If you don't know the answer, just say that you don't know. 
                    You may suggest non critical Healthcare issues even if the context is not in the pdf. 
                    Never Prescribe medications or become a substitute for doctor.
                    Do not make up an answer if you don't know about it.

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



            question = st.session_state.new_question

            answer = rag_chain.invoke(question)

            st.write(answer)




if __name__ == "__main__":
    main()



