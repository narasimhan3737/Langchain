import os
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from third_parties.linkedin import scrape_linkedin_profile



#information = 
"""
Barack Hussein Obama II[a] (born August 4, 1961) is an American politician who served as the 44th president of the United States from 2009 to 2017. As a member of the Democratic Party, he was the first African-American president in U.S. history. Obama previously served as a U.S. senator representing Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.

"""

if __name__ == "__main__":
    load_dotenv()
    #print("Hello World")
    #print(os.environ["OPENAI_API_KEY"])

    summary_template = """
    given the {information} about a person from I want you to create:
    1. Find who their spouse is?
    2. Create short summary on their spouse

    """

    summary_prompt_template = PromptTemplate(
        input_variables="information", template=summary_template
    )

    #llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #llm = ChatOllama(model="llama3")
    llm = ChatOllama(model="mistral")

    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data  = scrape_linkedin_profile(linkedin_profile_url="")

    res = chain.invoke(input={"information": linkedin_data})

    print(res)
