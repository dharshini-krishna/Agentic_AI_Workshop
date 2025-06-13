# travel_assistant_app.py

import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchRun, tool

# ========== Configuration ==========
GEMINI_API_KEY = "AIzaSyAV6HUVn4IgDMXgB86oOGsBWOjf-Y5J-kk"
WEATHER_API_KEY = "dc88cfe634fd43bb94984335251306"

# ========== Weather Tool ==========
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    temp_c = data['current']['temp_c']
    condition = data['current']['condition']['text']
    return f"The current temperature in {location} is {temp_c}°C with {condition}."

# ========== Search Tool ==========
search = DuckDuckGoSearchRun()

@tool
def get_attractions(city: str) -> str:
    """Get top tourist attractions in a given city."""
    results = search.run(f"Top tourist attractions in {city}")
    return results

# ========== LLM & Tools Setup ==========
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
tools = [get_weather, get_attractions]

# ========== REQUIRED Prompt Template ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Use the available tools to answer the user's questions."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ========== Agent Setup ==========
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ========== Streamlit Frontend ==========
st.set_page_config(page_title="Travel Assistant AI", page_icon="🌍")
st.title("🌍 Travel Assistant AI")
st.write("Get weather and top attractions for your travel destination.")

destination = st.text_input("Enter your travel destination:")

if st.button("Find Travel Info"):
    if destination:
        with st.spinner('Fetching travel information...'):
            try:
                result = agent_executor.invoke({"input": f"I am traveling to {destination}. Please tell me the weather and top attractions."})
                # Extract the final output from the returned dictionary
                final_response = result.get('output', 'Sorry, I could not find the answer.')
                st.success("Here’s what I found:")
                st.write(final_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a destination to proceed.")
