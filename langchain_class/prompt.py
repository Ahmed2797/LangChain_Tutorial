from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
import os 

parser = StrOutputParser()

llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.1',
                    task='text-generation',
                    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                    temperature=0.5) ## High temperture high creativate

model = ChatHuggingFace(llm=llm)


## 1.Simple_Prompt
template = PromptTemplate(
    template='You are a AI assistent in this topic {topic}',
    input_variables=['topic']
) 
prompt = template.invoke({'topic':'langchain'})


from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

## 2.system->-human
template = ChatPromptTemplate([
    ('system','You are an AI assistent in this domain {domain} expert'),
    ('human','Give me two important sentence about this topic {topic}')
]) 
prompt = template.invoke({'domain':'machine learining',
                          'topic':'RandomForest'})



## 2.1 message_prompt
message = [
    SystemMessage(content='You are a medical chatbot'),
    HumanMessage(content='Tell me about the Oncology decisis')
]
result = model.invoke(message)



## 3.chatbot
chat_history = [
    SystemMessage(content = 'You are a helpfull AI assistent')
]
while True:
    user_input = input['You': ]
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break 
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

print(chat_history)



## prompt_generator
prompt_generator = PromptTemplate(
    template=(
        "You are a medical chatbot.\n"
        "Now you can explain the topic: {topic}.\n"
        "Then you explain the disease: {disease}.\n"
        "Talk about the treatment: {treatment}.\n"
        "Finally, give some advice about {topic}."
    ),
    input_variables=["topic", "disease", "treatment"],
    validate_template=True
)




print(prompt)




