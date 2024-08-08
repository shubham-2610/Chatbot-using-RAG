from flask import Flask, request, render_template, redirect, url_for
import google.generativeai as genai
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

app = Flask(__name__)

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    pdf_loader = PyPDFLoader(temp_file_path)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 1})

    return vector_index

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.pdf'):
            vector_index = process_pdf(file)
            question = request.form.get('question')
            if question:
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=vector_index,
                    return_source_documents=True
                )
                result = qa_chain({"query": question})
                return render_template('index.html', result=result["result"], question=question)
            else:
                return render_template('index.html', error="Please enter a question.")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
