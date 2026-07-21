import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Initializing components")

embeddings = OpenAIEmbeddings(
        model="nomic-embed-text",
        openai_api_base="http://localhost:11434/v1", # The local endpoint
        openai_api_key="local-no-key-needed",        # Dummy bypass key
        check_embedding_ctx_length=False,             # Ollama expects string input, not pre-tokenized ints
)

llm = ChatAnthropic(model="claude-opus-4-8", max_tokens=4096)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based on the following context:
    
{context}

Question: {question}

Provide a detailed answer: """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    
    Limitations: 
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ===========================
# Implementation 2: With LCEL - Better Approach
# ===========================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL.
    Returns a chain that can be invoked with ("question": "...")

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy chain of operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving")

    # Query
    query = "what is Pinecone in machine learning ?"

    # ===========================
    # Option 0: Raw invocation without RAG
    # ===========================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ===========================
    # Option 1: Raw invocation without RAG
    # ===========================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ===========================
    # Option 2: Use implementation with LCEL
    # ===========================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming")
    print("- Built-in async")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
