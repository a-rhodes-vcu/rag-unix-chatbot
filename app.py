from flask import Flask, render_template, request

app = Flask(__name__)

def get_knowledge_base():
    from bs4 import BeautifulSoup
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    import requests
    import faiss

    url = 'https://mally.stanford.edu/~sr/computing/basic-unix.html'  # Replace with your target URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    plain_text = soup.get_text(strip=True)
    documents = plain_text.split(".")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    varPcb = [Document(page_content=text) for text in documents]

    varOcg = FAISS.from_documents(
        varPcb,
        embedding_model,
        distance_strategy="COSINE"
    )
    return varOcg

def chatbot_response(query_text):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    app.config["OPENAI_API_KEY"] = OPENAI_API_KEY

    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import RetrievalQA
    # Import PromptTemplate to define the custom prompt
    from langchain_core.prompts import PromptTemplate

    varOcg = get_knowledge_base()

    # 1. Define the Custom Prompt Template
    # Ensure {context} and {question} are present in the string
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    Helpful Answer:"""

    custom_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)

    # 3. Add the prompt to the chain via chain_type_kwargs
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=varOcg.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": custom_prompt}  # This injects your prompt
    )

    generated_answer = rag_chain.invoke(query_text)

    return str(generated_answer["result"])

@app.route("/")
def index():

    """render the html and css"""

    return render_template("index.html")
@app.route("/get")
def get_bot_response():

    """receive messages from the user and return messages from the bot"""

    msg = request.args.get("msg")  # get data from input
    res = chatbot_response(msg)
    return str(res)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
