# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:57:12 2024

@author: yash
"""

import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import pytube
from dotenv import load_dotenv
import os
load_dotenv()
## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title(" LangChain: Summarize Text From Youtube")
st.subheader('Summarize URL')

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 400 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])


# Check if URL is provided and valid
if generic_url:
    if validators.url(generic_url):
        try:
            # Attempt to load YouTube data using the provided URL
            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
            docs = loader.load()
            
            # Initialize the model
            llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

            # Create the prompt template
            prompt_template = """
            Provide a summary of the following content in 400 words:
            Content:{text}
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

            # Create and run the summarization chain
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)

            # Display the summary in Streamlit
            st.write(output_summary)
        
        except ValueError as ve:
            st.error(f"Error loading YouTube URL: {ve}")
    else:
        st.error("Please enter a valid URL.")
else:
    st.info("Please enter a YouTube or website URL to summarize.")