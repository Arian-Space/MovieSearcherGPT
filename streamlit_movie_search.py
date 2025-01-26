import streamlit as st
import json
import os
from langchain.retrievers import SelfQueryRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.query_constructor.base import load_query_constructor_runnable

def load_attribute_info():
    # Load attribute information for movie search
    with open('attribute_info.json', 'r') as file:
        return json.load(file)

def initialize_search_components(openai_api_key, elasticsearch_password):
    # Attribute information for movie search
    attribute_info = load_attribute_info()
    doc_contents = "Detailed description of the movie"

    # Initialize OpenAI components
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        api_key=openai_api_key, 
        temperature=0
    )

    # Create query constructor
    chain = load_query_constructor_runnable(llm, doc_contents, attribute_info)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Initialize Elasticsearch vector store
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id=os.environ['ES_CLOUD_ID'],
        index_name="sistem_index_movie",
        embedding=embeddings,
        es_user="elastic",
        es_password=elasticsearch_password,
    )

    # Create self-query retriever
    retriever = SelfQueryRetriever(
        query_constructor=chain,
        vectorstore=elastic_vector_search,
        verbose=True
    )

    return retriever

def main():

    st.set_page_config(
        page_title="Movie Searcher",
        page_icon="🍿",  # Puedes usar un emoji como icono o una URL de imagen
    )

    st.title("🎬 ChatGPT Movie Searcher 🔎")

    # API Key inputs
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.markdown(f"""<div style="text-align: justify;">Use OpenAI RAG capabilities to search for movies in the database from Elasticsearch.</div>""", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="text-align: justify;">
            The database we will use is <a href="https://huggingface.co/datasets/wykonos/movies">"wykonos/movies"</a> from huggingface 🤗, it is important to emphasize that to improve the performance of the program only 10 thousand movies are used in the database.
            </div>
            """,
            unsafe_allow_html=True)
        elasticsearch_password = os.environ['ELASTICSEARCHCLOUD_PAS']

    # Search input
    search_query = st.text_input("Enter your movie search query:")

    # Search button
    if st.button("Search in database"):
        # Validate inputs
        if not openai_api_key or not elasticsearch_password:
            st.error("Please provide an OpenAI API KEY to use the program.")
            return

        try:
            # Initialize search components
            retriever = initialize_search_components(
                openai_api_key, 
                elasticsearch_password
            )

            # Perform search
            results = retriever.invoke(search_query)

            # Display results
            st.subheader("Search Results")
            if results:
                for i, res in enumerate(results, 1):
                    # results[0].metadata["title"]
                    importantTitle = results[i-1].metadata["title"]

                    if results[i-1].metadata["tagline"] != "":
                        importantTitle = f"{importantTitle} ({results[i-1].metadata["tagline"]})"
                    
                    if results[i-1].metadata["vote_average"] != 0.0:
                        importantTitle = f"{importantTitle} ({results[i-1].metadata["vote_average"]})"
                    
                    with st.expander(importantTitle):

                        # Overview
                        if results[i-1].metadata["overview"] != "":
                            st.write(f"{results[i-1].metadata["overview"]}")
                        else:
                            st.write("No overview available.")
                        
                        st.subheader("Data about it:")

                        # genres
                        if results[i-1].metadata["genres"] != "":
                            st.write(f"Genres: {results[i-1].metadata["genres"]}.")

                        # runtime
                        if results[i-1].metadata["runtime"] != 0:
                            if results[i-1].metadata["runtime"] >= 60:
                                if results[i-1].metadata["runtime"]%60 != 0:
                                    st.write(f"Runtime of the movie: {int(results[i-1].metadata["runtime"]//60)}h and {int(results[i-1].metadata["runtime"]%60)}min.")
                                else:
                                    st.write(f"Runtime of the movie: {int(results[i-1].metadata["runtime"]//60)}h.")
                            else:
                                st.write(f"Runtime of the movie: {int(results[i-1].metadata["runtime"])} minutes.")

                        # languaje
                        if results[i-1].metadata["genres"] != "":
                            st.write(f"Original languaje: {results[i-1].metadata["original_language"]}.")

                        # release_date
                        if results[i-1].metadata["release_date"] != "":
                            st.write(f"Release date: {results[i-1].metadata["release_date"]}.")

                        # production_companies
                        if results[i-1].metadata["production_companies"] != "":
                            st.write(f"Production companies: {results[i-1].metadata["production_companies"]}.")

                        # budget
                        if results[i-1].metadata["budget"] != 0:
                            st.write(f"Budget of the movie: {results[i-1].metadata["budget"]}$.")

                        # revenue
                        if results[i-1].metadata["revenue"] != 0:
                            st.write(f"Revenue of the movie: {results[i-1].metadata["revenue"]}$.")
                        
                        # credits
                        if results[i-1].metadata["credits"] != "":
                            st.write(f"Credit's to: {results[i-1].metadata["credits"]}.")

                        # keywords
                        if results[i-1].metadata["keywords"] != "":
                            st.write(f"Keyword's used: {results[i-1].metadata["keywords"]}.")
            else:
                st.write("No results found.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()