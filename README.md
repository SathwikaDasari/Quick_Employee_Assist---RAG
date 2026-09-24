# Quick Employee Assist

An enterprise-style Employee Assist application built using
Retrieval-Augmented Generation (RAG) and Azure services.

## Architecture

User
  ↓
Streamlit UI
  ↓
RAG Orchestration
  ↓
Azure AI Search
  ↓
Relevant Document Chunks
  ↓
Azure OpenAI
  ↓
Grounded Answer
  ↓
User

## Azure Services Used

- Azure OpenAI
- Azure AI Search
- Azure Blob Storage
- Azure Cosmos DB

## AI Models

- text-embedding-3-small
- GPT model deployed through Azure OpenAI(gpt-5-mini)

## Features

- Upload and process employee policy documents
- PDF text extraction
- Document chunking
- Vector embeddings
- Vector search using Azure AI Search
- Context retrieval
- SLM-generated answers
- Source-grounded responses
- Chat history storage using Cosmos DB

## RAG Pipeline

1. Documents are uploaded to Blob Storage.
2. PDF content is extracted.
3. Documents are split into chunks.
4. Embeddings are generated.
5. Embeddings and document content are indexed in Azure AI Search.
6. User asks a question.
7. Azure AI Search retrieves relevant chunks.
8. Retrieved context is passed to Azure OpenAI.
9. The LLM generates a grounded response.
10. Conversation data is stored in Cosmos DB.

## Tech Stack

Python  
Streamlit  
Azure OpenAI  
Azure AI Search  
Azure Blob Storage  
Azure Cosmos DB  
RAG  
Vector Embeddings
