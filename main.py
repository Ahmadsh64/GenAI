import os
import google.generativeai as genai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Gemini Bot -AI-Powered Study Material Organizer and Summarizer using LLM API")

os.environ['GOOGLE_API_KEY'] = "<<<<<< Have your key here >>>>>>"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Select the model
model = genai.GenerativeModel('gemini-pro')

llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("Write a summary about War II")
print(result.content)



def create_output(input_text):
    output_text = """AI-Powered Study Material Organizer and Summarizer using LLM API
Input:
{}

Output:
AI-Powered Study Material Organizer and Summarizer using LLM API
""".format(input_text)
    return output_text


# Example input
input_text = ("In this project, you will be developing a Python application that utilizes the LLM API to help students "
              "organize and summarize their study materials. This application should be capable of labeling and categorizing"
              " different types of study materials (lecture notes, research articles, etc.) and provide concise summaries for quick revisions."
              " The system should also propose study plans or revision schedules based on the volume and complexity of the materials,"
              " demonstrating the LLM's capabilities of text categorization, summarization, and proposal generation.")

# Generate output
output_text = create_output(input_text)

# Print output
print(output_text)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "",
            "content": "Hello DR.Ethan : Give me topics to summarize"
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Process and store Query and Response
def llm_function(query):
    response = model.generate_content(query)

    # Displaying the Assistant Message
    with st.chat_message("assistant"):
        st.markdown(response.text)

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text
        }
    )


# Accept user input
query = st.chat_input("Message Gemini-Bot")

# Calling the Function when Input is Provided
if query:
    # Displaying the User Message
    with st.chat_message("user"):
        st.markdown(query)

    llm_function(query)