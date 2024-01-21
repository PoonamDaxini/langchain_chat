from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma 
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# chunk of 200 and then separator creates one docs
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts.txt")

docs = loader.load_and_split(
    text_splitter=text_splitter
)

# chroma is vector db -- nicely wrapped up in langchain
# persistent directory is sqllite db -- emb folder will be created
# embedding is done for each docs generated above
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# actual searching starts now
# k is no of result
#  on each run data get added in chrom db -- if we remove k=1 we can see multiple results - might be with same text content
results = db.similarity_search("what is an interseting fact about English Language?", k=1)


for result in results:
    print("\n")
    print(result.page_content)