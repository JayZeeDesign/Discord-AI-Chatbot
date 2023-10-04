import aiohttp
import io
from datetime import datetime
import time
import random
from urllib.parse import quote
from bot_utilities.config_loader import load_current_language, config
import openai
import os
from dotenv import find_dotenv, load_dotenv
import json

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from pydantic import Field
from langchain.prompts import ChatPromptTemplate
import requests


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()
current_language = load_current_language()
internet_access = config['INTERNET_ACCESS']

# openai.api_key = os.getenv('CHIMERA_GPT_KEY')
# openai.api_base = "https://api.naga.ac/v1"
def sdxl(prompt):
    response = openai.Image.create(
    model="sdxl",
    prompt=prompt,
    n=1,  # images count
    size="1024x1024"
)
    return response['data'][0]["url"]

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
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

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
    response = requests.post("https://chrome.browserless.io/content?token=0a049e5b-3387-4c51-ab6c-57647d519571", headers=headers, data=data_json)
    
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

    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': 'ab179d0f00ae0bafe47f77e09e62b9f53b3f281d',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()



def research(query):
    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You will always searching for internal knowledge base first to see if there are any relevant information
            2/ If the internal knowledge doesnt have good result, then you can go search online
            3/ While search online:
                a/ You will try to collect as many useful details as possible
                b/ If there are url of relevant links & articles, you will scrape it to gather more information
                c/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "system_message": system_message,
    }

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
        verbose=False,
        agent_kwargs=agent_kwargs,
    )

    results = agent.run(query)

    return results


def trigger_github_weekly_trending_repo_scrape():
    url = "https://api.browse.ai/v2/robots/0c0f94bf-207a-4660-8ade-238cd778bb25/tasks"

    payload = {"inputParameters": 
               {"originUrl": "https://github.com/trending"}
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

        # print(response)
        if response["statusCode"] == 200:
            if response["result"]["status"] == "successful":
                repos = response["result"]["capturedLists"]
                return repos                                 
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
    "🚀 Daily trending AI projects:

    **coqui-ai / TTS**
    - 🌟 3,952 stars this week | 18,952 total stars
    - 📖 a deep learning toolkit for Text-to-Speech, battle-tested in research and production
    - 🌐 https://github.com/coqui-ai/TTS

    **tldraw / tldraw**
    - 🌟 2,196 stars this week | 20,812 total stars
    - 📖 a very good whiteboard
    - 🌐 https://github.com/yoheinakajima/instagraph

    ...."

    if there is no any relevant repo, you can just say "Looks like no new interesting AI project today, let me know if I missed any pls!"
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | model

    results = chain.invoke({"repos": repos})

    return results.content

def generate_trending_git_report():
    repos = get_github_weekly_trending_repo()

    filtered_repos = filter_ai_github_repos(repos)

    return filtered_repos

    
async def fetch_models():
    return openai.Model.list()
    
agents = {}

def create_agent(id, user_name, ai_name, instructions):
    system_message = SystemMessage(
        content=instructions
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, ai_prefix=ai_name, user_prefix=user_name)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    tools = [                     
        Tool(
            name = "research",
            func = research,
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

    agents[id] = agent
    
    return agent


def generate_response(instructions, user_input):   
    id = user_input["id"]    
    message = user_input["message"]

    if id not in agents:
        user_name = user_input["user_name"]
        ai_name = user_input["ai_name"]
        agent = create_agent(id, user_name, ai_name, instructions)
    else:
        agent = agents[id]
    
    print(message)
    response = agent.run(message)

    return response


def generate_response_old(instructions, search, history):
    if search is not None:
        search_results = search
    elif search is None:
        search_results = "Search feature is disabled"
    messages = [
            {"role": "system", "name": "instructions", "content": instructions},
            *history,
            {"role": "system", "name": "search_results", "content": search_results},
        ]
    response = openai.ChatCompletion.create(
        model=config['GPT_MODEL'],
        messages=messages
    )
    message = response.choices[0].message.content
    return message


def generate_gpt4_response(prompt):
    messages = [
            {"role": "system", "name": "admin_user", "content": prompt},
        ]
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages
    )
    message = response.choices[0].message.content
    return message

async def poly_image_gen(session, prompt):
    seed = random.randint(1, 100000)
    image_url = f"https://image.pollinations.ai/prompt/{prompt}?seed={seed}"
    async with session.get(image_url) as response:
        image_data = await response.read()
        image_io = io.BytesIO(image_data)
        return image_io

# async def fetch_image_data(url):
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             return await response.read()

async def dall_e_gen(model, prompt, size, num_images):
    response = openai.Image.create(
        model=model,
        prompt=prompt,
        n=num_images,
        size=size,
    )
    imagefileobjs = []
    for image in response["data"]:
        image_url = image["url"]
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                content = await response.content.read()
                img_file_obj = io.BytesIO(content)
                imagefileobjs.append(img_file_obj)
    return imagefileobjs
    

async def generate_image_prodia(prompt, model, sampler, seed, neg):
    print("\033[1;32m(Prodia) Creating image for :\033[0m", prompt)
    start_time = time.time()
    async def create_job(prompt, model, sampler, seed, neg):
        if neg is None:
            negative = "(nsfw:1.5),verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.8),cross-eyed,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, repeating hair, nsfw, [[[[[bad-artist-anime, sketch by bad-artist]]]]], [[[mutation, lowres, bad hands, [text, signature, watermark, username], blurry, monochrome, grayscale, realistic, simple background, limited palette]]], close-up, (swimsuit, cleavage, armpits, ass, navel, cleavage cutout), (forehead jewel:1.2), (forehead mark:1.5), (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), multiple limbs, bad anatomy, (interlocked fingers:1.2),(interlocked leg:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, (deformed fingers:1.2), (long fingers:1.2)"
        else:
            negative = neg
        url = 'https://api.prodia.com/generate'
        params = {
            'new': 'true',
            'prompt': f'{quote(prompt)}',
            'model': model,
            'negative_prompt': f"{negative}",
            'steps': '100',
            'cfg': '9.5',
            'seed': f'{seed}',
            'sampler': sampler,
            'upscale': 'True',
            'aspect_ratio': 'square'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['job']
            
    job_id = await create_job(prompt, model, sampler, seed, neg)
    url = f'https://api.prodia.com/job/{job_id}'
    headers = {
        'authority': 'api.prodia.com',
        'accept': '*/*',
    }

    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(url, headers=headers) as response:
                json = await response.json()
                if json['status'] == 'succeeded':
                    async with session.get(f'https://images.prodia.xyz/{job_id}.png?download=1', headers=headers) as response:
                        content = await response.content.read()
                        img_file_obj = io.BytesIO(content)
                        duration = time.time() - start_time
                        print(f"\033[1;34m(Prodia) Finished image creation\n\033[0mJob id : {job_id}  Prompt : ", prompt, "in", duration, "seconds.")
                        return img_file_obj
