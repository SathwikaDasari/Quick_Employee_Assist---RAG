import streamlit as st
import os
import uuid
from datetime import datetime, timezone

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.cosmos import CosmosClient


# 1. Loading env variables

load_dotenv(override=True)


# Azure AI Search
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
search_index = os.getenv("AZURE_SEARCH_INDEX")


# Azure OpenAI
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")

chat_deployment = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT"
)

embedding_deployment = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "text-embedding-3-small"
)


# Cosmos DB
cosmos_endpoint = os.getenv(
    "AZURE_COSMOS_ENDPOINT"
)

cosmos_key = os.getenv(
    "AZURE_COSMOS_KEY"
)

cosmos_database = os.getenv(
    "AZURE_COSMOS_DATABASE"
)

cosmos_container = os.getenv(
    "AZURE_COSMOS_CONTAINER"
)


# Check required settings
required_settings = {
    "AZURE_SEARCH_ENDPOINT": search_endpoint,
    "AZURE_SEARCH_KEY": search_key,
    "AZURE_SEARCH_INDEX": search_index,

    "AZURE_OPENAI_ENDPOINT": openai_endpoint,
    "AZURE_OPENAI_KEY": openai_key,
    "AZURE_OPENAI_CHAT_DEPLOYMENT": chat_deployment,

    "AZURE_COSMOS_ENDPOINT": cosmos_endpoint,
    "AZURE_COSMOS_KEY": cosmos_key,
    "AZURE_COSMOS_DATABASE": cosmos_database,
    "AZURE_COSMOS_CONTAINER": cosmos_container
}


missing_settings = [
    key
    for key, value in required_settings.items()
    if not value
]


if missing_settings:

    st.error(
        "Missing configuration in .env: "
        + ", ".join(missing_settings)
    )

    st.stop()


# 2. Create Azure clients

search_client = SearchClient(

    endpoint=search_endpoint,

    index_name=search_index,

    credential=AzureKeyCredential(
        search_key
    )
)


openai_client = AzureOpenAI(

    azure_endpoint=openai_endpoint,

    api_key=openai_key,

    api_version="2024-10-21"
)


cosmos_client = CosmosClient(

    cosmos_endpoint,

    cosmos_key
)


database = cosmos_client.get_database_client(

    cosmos_database
)


container = database.get_container_client(

    cosmos_container
)


# 3. Get conversation history

def get_chat_history(session_id):

    query = """
    SELECT c.question, c.answer, c.sources, c.timestamp
    FROM c
    WHERE c.sessionId = @sessionId
    ORDER BY c.timestamp ASC
    """

    parameters = [

        {
            "name": "@sessionId",

            "value": session_id
        }

    ]


    items = container.query_items(

        query=query,

        parameters=parameters,

        enable_cross_partition_query=False
    )


    return list(items)


# 4. Saving conversation to Cosmos DB

def save_chat_record(

    session_id,

    question,

    answer,

    sources
):

    chat_record = {

        "id": str(uuid.uuid4()),

        "sessionId": session_id,

        "question": question,

        "answer": answer,

        "sources": sources,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


    container.create_item(

        body=chat_record
    )


# 5. RAG Pipeline

def ask_rag(

    question,

    session_id
):

  #conversation History

    history = get_chat_history(

        session_id
    )


    conversation_history = ""


    for chat in history[-4:]:

        conversation_history += f"""
Previous User Question:
{chat.get("question", "")}

Previous Assistant Answer:
{chat.get("answer", "")}

"""


    # Createing embedding

    query_response = openai_client.embeddings.create(

        model=embedding_deployment,

        input=question
    )


    query_vector = (
        query_response.data[0].embedding
    )


    #Azure AI Search - Hybrid Search

    vector_query = VectorizedQuery(

        vector=query_vector,

        k_nearest_neighbors=5,

        fields="contentVector"
    )


    results = search_client.search(

        # Keyword search
        search_text=question,

        # Vector search
        vector_queries=[
            vector_query
        ],

        select=[
            "content",
            "source",
            "page"
        ],

        top=5
    )


    results = list(results)



    # Building document context

    context = ""


    for i, result in enumerate(results):

        context += f"""
CHUNK {i}

Source:
{result.get("source", "Unknown")}

Page:
{result.get("page", "Unknown")}

Content:
{result.get("content", "")}

"""


    # No documents retrieved

    if not context.strip():

        answer = (
            "I couldn't find that information "
            "in the provided documents."
        )


        sources = []


        save_chat_record(

            session_id,

            question,

            answer,

            sources
        )


        return answer, sources


    # RAG Prompting

    prompt = f"""
You are Quicky Employee Assist, an internal employee information assistant.

Answer the CURRENT USER QUESTION using ONLY the provided conversation
history and retrieved document context.

Rules:

1. The CURRENT USER QUESTION is the primary question you must answer.

2. Use the retrieved document context as the factual source.

3. Perform the answer based on the documents retrieved for the
   current question.

4. Use conversation history ONLY when the current question is a
   follow-up that depends on something previously discussed.

5. Do not let previous questions or answers override the retrieved
   documents for the current question.

6. Do not use outside knowledge.

7. Do not invent or assume information.

8. If the answer is present in the retrieved context, answer it directly.

9. If the answer is not available in the retrieved context, respond exactly:

"I couldn't find that information in the provided documents."

10. Keep the answer concise and professional.

11. If the answer contains multiple relevant points, use bullet points.

12. Only include information directly relevant to the user's question.

13. Do not add unrelated missing-information statements.

14. Never mention chunk numbers such as CHUNK 0, CHUNK 1, CHUNK 2,
    or similar internal retrieval information.

15. Do not mention embeddings, vector search, RAG, prompts,
    or other internal implementation details.

16. Do not return JSON.

17. Return only the answer to the CURRENT USER QUESTION.

Conversation history:

{conversation_history if conversation_history else "No previous conversation."}


Retrieved document context:

{context}


CURRENT USER QUESTION:

{question}


Answer:
"""


    # Azure OpenAI

    response = openai_client.chat.completions.create(

        model=chat_deployment,

        messages=[

            {
                "role": "system",

                "content": (
                    "You are Quicky Employee Assist. "
                    "Answer questions only using the "
                    "provided conversation history "
                    "and document context."
                )
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        max_completion_tokens=1000
    )


    answer = response.choices[0].message.content


    if not answer:

        answer = (
            "I couldn't find that information "
            "in the provided documents."
        )


    answer = answer.strip()


    # Source handling
  
    sources = []


    fallback_message = (
        "I couldn't find that information "
        "in the provided documents."
    )


    if answer.strip() != fallback_message:

        seen_sources = set()


        for result in results:

            source = result.get(
                "source"
            )

            page = result.get(
                "page"
            )


            if not source:
                continue


            source_key = (

                source,

                page

            )


            if source_key not in seen_sources:

                sources.append({

                    "source": source,

                    "page": page

                })


                seen_sources.add(
                    source_key
                )


        # Display maximum 3 unique sources
        sources = sources[:3]


    # Save conversation

    save_chat_record(

        session_id,

        question,

        answer,

        sources
    )


    return answer, sources


# 6. Streamlit UI


st.set_page_config(

    page_title="Quicky Employee Assist",

    page_icon="📄",

    layout="centered"
)


st.title(
    "📄 Quicky Employee Assist"
)


st.write(
    "Ask questions about company policies and procedures."
)


# 7. Create session


if "session_id" not in st.session_state:

    st.session_state.session_id = str(
        uuid.uuid4()
    )


# 8. Store UI messages


if "messages" not in st.session_state:

    st.session_state.messages = []


# 9. Display previous messages


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# 10. User Question


question = st.chat_input(
    "Ask your question..."
)


if question:

   

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    with st.chat_message("user"):

        st.write(
            question
        )

    #Run RAG


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching employee documents..."
        ):

            try:

                answer, sources = ask_rag(

                    question,

                    st.session_state.session_id
                )



                st.write(
                    answer
                )



                if sources:

                    st.markdown(
                        "**📄 Sources**"
                    )


                    for source in sources:

                        page = source.get(
                            "page"
                        )


                        if page is not None:

                            st.caption(

                                f"📄 {source['source']} "
                                f"— Page {page}"

                            )

                        else:

                            st.caption(

                                f"📄 {source['source']}"

                            )


               #UI

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })


            except Exception as e:

                st.error(
                    "Something went wrong "
                    "while processing your question."
                )

                st.exception(e)