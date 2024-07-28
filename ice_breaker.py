import os
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == '__main__':
    print("Hello, LangChain!")
    print(os.getenv('OPENAI_API_KEY'))
    print(os.environ['OPENAI_API_KEY'])
    
    
    # https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html
    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    # https://python.langchain.com/v0.1/docs/modules/model_io/chat/quick_start/
    # https://openai.com/api/pricing/
    llm = ChatOpenAI(temperature=0, name="gpt-4o-mini")
    
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
    )
    res = chain.invoke(input={"information": linkedin_data})

    print(res['text'])