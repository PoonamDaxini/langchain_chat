from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()


embeddings = OpenAIEmbeddings()
# read from emb -- no writes
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantRetriever(
    embeddings=embeddings,
    chroma=db
    )
chat = ChatOpenAI()
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What is an intersting fact about an english language?")

print(result)