from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document


class RedundantRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma
    def get_relevant_documents(self, query):
        # calculate embeddings for query string
        # one way

        # embeddings = OpenAIEmbeddings()
        # emb = embeddings.embed_query(query)

        emb = self.embeddings.embed_query(query)

        #  take embedding and feed them to max_relavance function
        # chroma instance with persistant directory
        #  to avoid hard codding will take that as class object (instantiate while using it)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        return []