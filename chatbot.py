import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = OpenAI(model='gpt-3.5-turbo-instruct',
             temperature=0.7)

ice_cream_assistant_template = """
You are an ice cream assistant chatbot named "Scoopsie". Your expertise is 
exclusively in providing information and advice about anything related to ice creams. This includes flavor combinations, ice cream recipes, and general 
ice cream-related queries. You do not provide information outside of this 
scope. If a question is not about ice cream, respond with, "I specialize only in ice cream related queries." 
Question: {question} 
Answer:"""

ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=ice_cream_assistant_template
)
llm_chain = LLMChain(llm=llm, prompt=ice_cream_assistant_prompt_template)

def query_llm(question):
    print(llm_chain.invoke({'question': question})['text'])

if __name__ == '__main__':
    query_llm("Who are you?")