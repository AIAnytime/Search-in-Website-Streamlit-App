import streamlit as st 
from urllib.parse import urlparse
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import openai
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from langchain.tools.base import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
import os, asyncio, trafilatura
from langchain.docstore.document import Document
import requests

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )

class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information and answers relevant to the question. Please use bullet points to list the answers"
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        response = requests.get(url)
        page_content = response.text
        print(page_content)
        docs = [Document(page_content=page_content, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        print(results_docs)
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError

def run_llm(url, query):
    llm = ChatOpenAI(temperature=0.5)
    query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))
    result = query_website_tool._run(url, query)  # Pass the URL and query as arguments
    return result

st.markdown("<h1 style='text-align: center; color: green;'>Info Retrieval from Website 🦜 </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: green;'>Developed by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color:red;'>Enter the Website URL 👇</h2>", unsafe_allow_html=True)

input_url = st.text_input("Enter the URL")

if len(input_url)>0:
    url_name = get_url_name(input_url)
    st.info("Your URL is: 👇")
    st.write(url_name)

    st.markdown("<h4 style='text-align: center; color:green;'>Enter Your Query 👇</h4>", unsafe_allow_html=True)
    your_query = st.text_area("Query the Website")
    if st.button("Get Answers"):
        if len(your_query)>0:
            st.info("Your query is: "+ your_query)

            final_answer = run_llm(input_url, your_query)
            st.write(final_answer)

