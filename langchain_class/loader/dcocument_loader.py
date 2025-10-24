from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
import os 

str_parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.1',
                    task='text-generation',
                    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                    temperature=0.5) 

model = ChatHuggingFace(llm=llm)
template = PromptTemplate(
    template='Explain the topic {topic}.\n\n Give 5 line of summary about \n{topic}',
    input_variables=['topic']
)  




## Text_loader 
from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path='book.txt',encoding='utf-8')
doc = loader.load()
chain = template | model | str_parser
print(chain.invoke({'topic':'langchain'}))
print(chain.invoke({'topic':doc[0].page_content}))




## CSV_file loader
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Books.csv',autodetect_encoding=True)
doc  =loader.load() 
print(doc[1]) 



## pdf_loader 
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path='Books.pdf',autodetect_encoding=True)
doc  =loader.load() 
print(doc[0].page_content)
print(doc[1].metadata)




## Directory_loader 
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(path='Books',
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)
                         
doc  =loader.load() 
print(doc[0].page_content)
print(doc[1].metadata)




## webBaseloader 
from langchain_community.document_loaders import WebBaseLoader

web_template = PromptTemplate(
    template='Give the information of this product {product}.\n\n Give 5 line of summary about \n{text}',
    input_variables=['product','text']
) 
url = ''
loader = WebBaseLoader(web_path=url)
doc  =loader.load() 
chain = web_template | model | str_parser
result = chain.invoke({'product':'FaceMask','text':doc[1].metadata})

print(doc[0].page_content)
print(doc[1].metadata)



## Dccument_loader
#  https://python.langchain.com/docs/integrations/document_loaders/

