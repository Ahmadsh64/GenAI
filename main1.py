import os
import google.generativeai as genai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from io import BytesIO

# Load environment variables from a .env file
load_dotenv()

# Configure API keys
os.environ['GOOGLE_API_KEY'] = "<<<<<< Have your key here >>>>>>"
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Validate that GOOGLE_API_KEY is loaded
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable in the .env file.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

    # Select the model
    model_name = 'gemini-pro'
    model = genai.GenerativeModel(model_name)
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)

    def categorize_material(text):
        response = llm.invoke(f"Categorize the following text: {text}")
        return response.content if response else "No category available."

    def summarize_text(text):
        response = llm.invoke(f"Summarize the following text: {text}")
        return response.content if response else "No summary available."

    def create_study_plan(text, language):
        response = llm.invoke(f"Create a study plan based on the following text in {language} language: {text}")
        return response.content if response else "No study plan available."

    st.title("Gemini Bot - AI-Powered Study Material Organizer and Summarizer using LLM API")

    # UI for uploading files
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

    # Select language
    language = st.selectbox("Language for summarization", ["hebrew", "english", "arabic"], index=0)

    if uploaded_file:
        file_content = None
        file_type = uploaded_file.type

        # Detect file type and read content accordingly
        if file_type == "text/plain":
            file_content = uploaded_file.read().decode('utf-8', errors='replace')
        elif file_type == "application/pdf":
            try:
                # קריאה בקובץ PDF
                pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
                file_content = "\n".join(page.extract_text() for page in pdf_reader.pages)
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = docx.Document(BytesIO(uploaded_file.read()))
                file_content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            except Exception as e:
                st.error(f"Error reading DOCX file: {e}")

        if file_content:
            st.write("Uploaded content:", file_content)

            if st.button("Process"):
                # Process the uploaded content
                category = categorize_material(file_content)
                summary = summarize_text(file_content)
                study_plan = create_study_plan(file_content, language)

                # Display the results in three separate columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Category")
                    st.write(category)

                with col2:
                    st.subheader("Summary")
                    st.write(summary)

                with col3:
                    st.subheader("Study Plan")
                    st.write(study_plan)

    else:
        # UI for direct text input
        user_input = st.text_area("Enter text to process")

        if user_input:
            if st.button("Process"):
                # Process the direct input
                category = categorize_material(user_input)
                summary = summarize_text(user_input)
                study_plan = create_study_plan(user_input, language)

                # Display the results in three separate columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Category")
                    st.write(category)

                with col2:
                    st.subheader("Summary")
                    st.write(summary)

                with col3:
                    st.subheader("Study Plan")
                    st.write(study_plan)


    def create_output(input_text):
        output_text = f"""AI-Powered Study Material Organizer and Summarizer using LLM API
        Input:
        {input_text}

        Output:
        AI-Powered Study Material Organizer and Summarizer using LLM API
        """
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
                "role": "assistant",
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

        # Storing the Assistant Message
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
