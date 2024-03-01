import json
import logging
import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import asyncio
import threading


def analyze_risk(parsed_guidlines, parsed_application_form):
    prompt = f"""<guidelines>{parsed_guidlines}</guidelines> <application_form>{parsed_application_form}</application_form>."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "You will be provided an insurance underwriting guidelines (delimited with XML tags) <guidelines></guidelines> and an insurance application form (delimited with XML tags) <application_form></application_form>. Can you provide some information about the relevant risks on the application following the guidelines ?."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    relevant_risks = []
    for choice in response.choices:
        relevant_risks.append(choice.message.content)

    return relevant_risks


@st.cache_data
@st.cache_resource
def extract_data(file, file_name):
    if isinstance(file, bytes):  
        with open("temp_file.pdf", "wb") as temp_file:
            temp_file.write(file)
        file_path = "temp_file.pdf"
    else:
        file_path = file.name

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if file_name == "underwriting_guidlines":
        text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=50, 
            model_name="BAAI/bge-large-en-v1.5", 
            tokens_per_chunk=512
        )
        guidelines_docs = text_splitter.split_documents(pages)
        guidelines = "\n\n".join([doc.page_content for doc in guidelines_docs])
        chunk_size = 512
        summarized_chunks = []
        chunks = [guidelines[i:i+chunk_size] for i in range(0, len(guidelines), chunk_size)]
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        for chunk in chunks:
            summarized_chunk = summarizer(chunk, max_length=chunk_size)
            summarized_chunks.append(summarized_chunk)
        summarized_texts = [result[0]["summary_text"] for result in summarized_chunks]
        text_content = "\n\n".join(summarized_texts)
    
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, length_function=len)
        docs = text_splitter.split_documents(pages)
        text_content = "\n\n".join([doc.page_content for doc in docs])

    if isinstance(file, bytes):
        os.remove(file_path)

    return text_content



async def extract_data_async(file, file_name):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_data, file, file_name)




async def main():
    st.title('Risk Analysis')

    st.write('Upload files for risk analysis:')
    uploaded_underwriting_guidelines = st.file_uploader('Underwriting Guidelines', type="pdf", help='Upload Insurance underwriting guidelines')
    uploaded_application_form = st.file_uploader('Application form', type="pdf", help='Upload Insurance application form')
        
    if st.button('Analyze Risk'):
        with st.spinner("Processing files..."):
            parsed_guidelines = extract_data(uploaded_underwriting_guidelines, "underwriting_guidlines")
            parsed_application_form = extract_data(uploaded_application_form, "application_form")
 
        # with st.spinner("Processing files..."):
        #     parsed_guidelines = await extract_data_async(uploaded_underwriting_guidelines, "underwriting_guidelines")
        #     parsed_application_form = await extract_data_async(uploaded_application_form, "application_form")
            
        
        if parsed_guidelines is not None and parsed_application_form is not None:       
            with st.spinner("Analyzing risk..."):     
                relevant_risks = analyze_risk(parsed_guidelines, parsed_application_form)
                st.write('Relevant Risks :')
                for risk in relevant_risks:
                    st.write(risk)
        
        else:
            st.write('Please upload both files to analyze risk.')


            
if __name__ == '__main__':
    asyncio.run(main())
