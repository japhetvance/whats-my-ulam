import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone as PineconeClient
import nltk

# Initialize session state variables if they don't exist
if "loads" not in st.session_state:
    st.session_state.loads = False

if not st.session_state.loads:
    load_dotenv()
    nltk.download('punkt')
    st.session_state.loads = True

# Retrieve environment variables
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
openai_api_key = os.environ.get("OPENAI_API_KEY")

openai.api_key = openai_api_key
pc = PineconeClient(api_key=pinecone_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
index = pc.Index(pinecone_index_name)

# Cache the embeddings to avoid redundant API calls
@st.cache_data(show_spinner=False)
def get_query_embedding(query):
    return openai.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

def generate_response(query, score_threshold=0.75):
    query_embedding = get_query_embedding(query)
    results = index.query(vector=query_embedding, top_k=11, include_metadata=True)
    filtered_results = [match for match in results['matches'] if match['score'] >= score_threshold]
    return [{
        'Dish': match['id'],
        'Summary': match['metadata']['Summary'],
        'Ingredients': match['metadata']['Ingredients'],
        'Description': match['metadata']['Description'],
        'Score': match['score'],
    } for match in filtered_results]

def main():
    st.title("What's My Ulam Pare?")
    # Input field for the dish description
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("",placeholder="Describe your ulam pare!")
    with col2:
        button_html = """
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            padding: 8px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 12px;
        }
        div.stButton > button:hover {
            color: white;
            background-color: #45a049;
        }
        div.stButton > button:active {
            color: white;
        }
        </style>
        """
        st.markdown(button_html, unsafe_allow_html=True)  # Injects the custom style

        if st.button("Search"):
            if query:
                response = generate_response(query)
                st.session_state.response = response  # Store response in session state
            else:
                st.session_state.response = None

    # with col3:
    #     if st.button("Clear"):
    #         st.session_state.response = None
    #         query = "" # Clear response on button click
    #         st.rerun()  # Rerun the app to reset the input field

    # Display results if available
    # Display results if available
    if 'response' in st.session_state:
        if st.session_state.response:
            response = st.session_state.response
            # Display the main dish information
            st.markdown("---")
            st.markdown(f"### Pare, your ulam is <span style='color:#d3806f'>{response[0]['Dish']}</span>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### Description:")
            st.write(response[0]['Description'])
            # Split ingredients by comma and format as unordered list
            st.markdown("---")
            ingredients = response[0]['Ingredients'].split(',')
            st.markdown("#### Ingredients:")
            st.markdown("\n".join([f"- {ingredient.strip()}" for ingredient in ingredients]))
            st.markdown("---")
            st.markdown("#### Summary:")
            st.write(response[0]['Summary'])
            st.markdown("---")
            
            # Display similar dishes as clickable items
            st.markdown("#### Similar Dishes")
            for dish in response[1:]:
                with st.expander(dish['Dish'], expanded=False):
                    st.markdown(f"#### {dish['Dish']}")
                    st.markdown("##### Description")
                    st.write(dish['Description'])
                    st.markdown("##### Ingredients")
                    st.write(dish['Ingredients'])
                    st.markdown("##### Summary")
                    st.write(dish['Summary'])
        else:
            st.markdown("### No results found")
            st.write("Please provide a more accurate dish description or include ingredients for better results.")

if __name__ == "__main__":
    main()