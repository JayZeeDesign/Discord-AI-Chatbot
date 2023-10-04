from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
from langchain.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

def knowledge_retrieval(query):    
    # Define the data to be sent in the request
    data = {
        "params":{
            "query":query
        },
        "project": "feda14180b9d-4ba2-9b3c-6c721dfe8f63"
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post("https://api-1e3042.stack.tryrelevance.com/latest/studios/6eba417b-f592-49fc-968d-6b63702995e3/trigger_limited", data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        return response.json()["output"]["answer"]
    else:
        print(f"HTTP request failed with status code {response.status_code}") 

def summary(content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    output = summary_chain.run(input_documents=docs,)

    return output


def scrape_website(url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url        
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post("https://chrome.browserless.io/content?token=2db344e9-a08a-4179-8f48-195a2f7ea6ee", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")     


def search(query):
    """
    Asynchronously searches for a prompt and returns the search results as a blob.

    Args:
        prompt (str): The prompt to search for.

    Returns:
        str: The search results as a blob.

    Raises:
        None
    """

    endpoint = "https://ddg-api.herokuapp.com/search"
    params = {
        'query': query,  # Replace with your search query
        'limit': 5  # Replace with your desired limit
    }
    
    # Make the GET request
    response = requests.get(endpoint, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()
        return results
    else:
        return (f"Didn't get any results")

def research(query):
    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You will always searching for internal knowledge base first to see if there are any relevant information
            2/ If the internal knowledge doesnt have good result, then you can go search online
            3/ While search online or scrape website, you should follow these steps:
                a/ You will try to collect as many useful details as possible; 
                b/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "system_message": system_message,
    }

    memory = ConversationBufferWindowMemory(return_messages=True)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [        
        Tool(
            name="Knowledge_retrieval",
            func=knowledge_retrieval,
            description="Use this to get our internal knowledge base data for curated information, always use this first before searching online"
        ),      
        Tool(
            name = "Google_search",
            func = search,
            description = "Always use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),           
        Tool(
            name = "Scrape_website",
            func = scrape_website,
            description = "Use this to load content from a website url"
        ),   
    ]

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory
    )

    results = agent.run(query)

    return results


# response = research("Give me a summary of trending AI projects on github (https://github.com/trending)")
# print(response)
def trigger_github_weekly_trending_repo_scrape():
    url = "https://api.browse.ai/v2/robots/0c0f94bf-207a-4660-8ade-238cd778bb25/tasks"

    payload = {"inputParameters": 
               {"originUrl": "https://github.com/trending",
                "weekly_trending_github_repo_limit": 10}
            }
    headers = {"Authorization": "Bearer ec2cc08b-3343-47c9-9dd3-dc5d40d4aa3b:dead067b-d485-496d-a3e0-4902339f6cfe"}

    response = requests.request("POST", url, json=payload, headers=headers)

    print("id: ", response.json()["result"]["id"], "is :", response.text)
    return response.json()["result"]["id"]

def retrieve_github_weekly_trending_repo(task_id):
    url = f"https://api.browse.ai/v2/robots/0c0f94bf-207a-4660-8ade-238cd778bb25/tasks/{task_id}"

    headers = {"Authorization": "Bearer ec2cc08b-3343-47c9-9dd3-dc5d40d4aa3b:dead067b-d485-496d-a3e0-4902339f6cfe"}

    response = requests.request("GET", url, headers=headers)

    return response.json()

def get_github_weekly_trending_repo():
    task_id = trigger_github_weekly_trending_repo_scrape()    

    while True:
        time.sleep(5)

        response = retrieve_github_weekly_trending_repo(task_id)
        print(response)

        # print(response)
        if response["statusCode"] == 200:
            if response["result"]["status"] == "successful":
                repos = response["result"]["capturedLists"]["weekly trending github repo"]
                processed_repos = []
                for repo in repos:
                    status = repo.get("_STATUS")
                    if status != "REMOVED":
                        # Return everything except _STATUS
                        processed_repos.append(repo)                    
                print(processed_repos)
                return processed_repos
            elif response["result"]["status"] == "failed":
                return "failed to get data"
        elif response["statusCode"] in {400, 401, 403, 404, 500, 503}:
            return response["messageCode"]

def filter_ai_github_repos(repos):
    model = ChatOpenAI()

    prompt_template = """
    {repos} 
    Above is the list of scraped trending github repos this week, 
    can you help me filter out ones that is related to AI, knowledge graph, computer vision, large language model?

    The report should be in certain format:
    "🚀 Weekly trending AI projects:

    coqui-ai / TTS
    - 🌟 3,952 stars this week | 18,952 total stars
    - 📖 a deep learning toolkit for Text-to-Speech, battle-tested in research and production
    - 🌐 https://github.com/coqui-ai/TTS

    tldraw / tldraw
    - 🌟 2,196 stars this week | 20,812 total stars
    - 📖 a very good whiteboard
    - 🌐 https://github.com/yoheinakajima/instagraph

    ...."
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | model

    results = chain.invoke({"repos": repos})

    return results.content

def generate_trending_git_report():
    repos = get_github_weekly_trending_repo()

    filtered_repos = filter_ai_github_repos(repos)

    return filtered_repos
