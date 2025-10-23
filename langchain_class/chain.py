from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import Field,BaseModel 
from typing import Literal
import os 

str_parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.1',
                    task='text-generation',
                    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                    temperature=0.5) ## High temperture high creativate

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='You are a AI assistent in this topic {topic}',
    input_variables=['topic']
) 

template1 = PromptTemplate(
    template='write the 5 line summary in text {text}',
    input_variables=['text']
) 
## simple_chain
prompt = template.invoke({'topic':'langchain'}) 
result = prompt | model | str_parser  

prompt2 = template.invoke({'text':result.content}) 
result2 = prompt2 | model | str_parser

## sequential_chain
result2 = prompt | model | str_parser | prompt2 | model | str_parser
print(result2.content) 



## paraell_chain
str_parser = StrOutputParser()
model = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm) # model will changes or try different model

template = PromptTemplate(
    template='Generate short and simple notes in the following text {text}',
    input_variables=['text']
) 

template1 = PromptTemplate(
    template='Generate some important question in this {text}',
    input_variables=['text']
) 

template2 = PromptTemplate(
    template='Merge the provided text into a single document with notes ->{notes} and example -> {example}\n ',
    input_variables=['notes','example']
) 
## paraell_chain
paraell_chain = RunnableParallel({
    'notes': template | model | str_parser,
    'example': template1 | model | str_parser
})
merge_chain = template2 | model | str_parser

chain = paraell_chain | merge_chain 

text = '''
Machine Learning (ML) is a branch of Artificial Intelligence (AI) that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

It is mainly divided into three categories:

Supervised Learning:
The model learns from labeled data.
Example — Predicting house prices using previous house data (price, area, location, etc.).

Unsupervised Learning:
The model finds hidden patterns in unlabeled data.
Example — Customer segmentation based on purchasing behavior.

Reinforcement Learning:
The model learns by interacting with the environment and receiving rewards or penalties.
Example — Teaching a robot to walk or play chess.
'''
print(chain.invoke({'text':'text'}))



## condition_chain 
class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

template = PromptTemplate(
    template='classify the following text feedback positive or negative into text {feedback}\n {format_instractions}',
    input_variables=['feedback'],
    partial_variables={'format_instractions':pydantic_parser.get_format_instructions()}
)
classify_chain = template | model | pydantic_parser

template2 = PromptTemplate(
    template= 'Get the positve feedback {feedback}',
    input_variables=['feedback']
)
template3 = PromptTemplate(
    template= 'Get the negative feedback {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch([
    (lambda x:x.sentiment=='positive',template2 | model | str_parser),
    (lambda x:x.sentiment=='negative',template3 | model | str_parser),
    RunnableLambda(lambda x: 'Error sentiment')
    
])

chain = classify_chain | branch_chain
print(chain.invoke({'feedback':'Iphone is very cheap'}))





