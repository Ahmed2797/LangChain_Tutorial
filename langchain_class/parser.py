from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
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
prompt = template.invoke({'topic':'langchain'}) 
result = prompt | model | str_parser  

prompt2 = template.invoke({'text':result.content}) 
result2 = prompt2 | model | str_parser

## sequential_chain
result2 = prompt | model | str_parser | prompt2 | model | str_parser
print(result2.content) 



## Jsonoutputparser 
json_parser = JsonOutputParser()

template = PromptTemplate(
    template='You are a AI assistent in this topic {topic}\n {format_instractions}',
    input_variables=['topic'],
    partial_variables={'format_instractions':json_parser.get_format_instructions()}
) 
## simple_chain
chain = template | model | json_parser
print(chain.invoke({'topic':'langchain'})) 



## pydanticoutputparser 
pydantic_parser = PydanticOutputParser()

class person(BaseModel):
    name:str = Field(discriminator='name of the person')
    age:int = Field(discriminator='age of the person')
    city:str = Field(description='name of the city person belongs to')

template = PromptTemplate(
    template='Generate name,age,city of the person in {place} palce \n {format_instractions}',
    input_variables=['place'],
    partial_variables={'format_instractions':pydantic_parser.get_format_instructions()}
)
chain = template | model | pydantic_parser
print(chain.invoke({'place':'Narsingdi'})) 



## StructuredOutputParser
schema = [
    ResponseSchema(name='fact1',description='fact1 about the topic'),
    ResponseSchema(name='fact2',description='fact2 about the topic'),
    ResponseSchema(name='fact3',description='fact3 about the topic')
]
structure_parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':structure_parser.get_format_instructions()}
)
chain = template | model | structure_parser
print(chain.invoke({'topic':'black hole'}))



