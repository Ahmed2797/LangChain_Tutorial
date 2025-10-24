from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import Chroma,FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.chains import retrieval_qa

from langchain_openai import ChatOpenAI



## WikipediaRetriever
wiki_retriever = WikipediaRetriever()
result = wiki_retriever.get_relevant_documents("Support Vector Machine")

for doc in result:
    print(doc.page_content[:300]) 




## victor_store_retriver -----> Similarity Search -----> Clossed_Victor_directions
vectorstore = Chroma.from_documents(
    documents=None,
    embedding= None,
    collection_name='My_Collection'
)
vs_retriver = vectorstore.as_retriever(search_kwargs={"k": 2})




## MMR
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",                   
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)




## Multiquery Retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)




## ContextualCompressionRetriever
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)
contextual_compression_retriever = ContextualCompressionRetriever.from_llm(
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    base_compressor=compressor
)




## result_try all_retriver
qa_chain = retrieval_qa.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vs_retriver ## change the retriver trial(example=vs_retriver,mmr_retriever)
)
result = qa_chain.run("What is LangChain?")
print(result)







 





##  https://python.langchain.com/docs/integrations/retrievers/



# LangChain Community Retrievers সাধারণত ব্যবহার হয় যেখানে
# external knowledge retrieval দরকার হয় — যেমনঃ

# Chatbot (যা ইন্টারনেট বা Wikipedia থেকে উত্তর দেয়) ## WikipediaRetriever

# Research assistant (Arxiv থেকে paper summary আনে) ## ArxivRetriever

# Code assistant (GitHub থেকে কোড খোঁজে) ## GitHubRetriever

# Video summarizer (YouTube থেকে ভিডিও transcript আনে) ## YouTubeRetriever

