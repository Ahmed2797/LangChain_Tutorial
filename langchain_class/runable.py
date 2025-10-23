from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import (RunnableParallel,RunnableSequence,RunnableBranch,
                                      RunnablePassthrough,RunnableLambda)
from pydantic import Field,BaseModel 
from typing import Literal
import os 

str_parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.1',
                    task='text-generation',
                    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                    temperature=0.5) ## High temperture high creativate


## RunnableParallel
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
## RunnableParallel
paraell_chain = RunnableParallel({
    'notes': template | model | str_parser,
    'example': template1 | model | str_parser
})

## RunnableSequence
paraell_chain = RunnableParallel({
    'notes': RunnableSequence(template, model, str_parser),
    'example': RunnableSequence(template1, model, str_parser)
})

merge_chain = template2 | model | str_parser

chain = paraell_chain | merge_chain 
text = '''
...............................
''' 
print(chain.invoke({'text':'text'}))


## RunnableBranch 
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




