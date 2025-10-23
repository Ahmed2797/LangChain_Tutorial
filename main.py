import os
from dotenv import load_dotenv 
load_dotenv()
hug = os.getenv('HUGGINGFACEHUB_API_TOKEN')
print(hug)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence
import torch

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",   # Free model
    huggingfacehub_api_token=hug
    #huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Define the prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Explain the topic: {topic}")

# Define the output parser
parser = StrOutputParser()

# Build the chain (modern LCEL style)
chain = prompt | llm | parser

# Run it
result = chain.invoke({"topic": "Machine Learning"})
print(result)


