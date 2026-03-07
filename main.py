import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


print("Initializing components...")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=1536)
llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ['INDEX_NAME']
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Adjust 'k' to retrieve more or fewer relevant documents

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ========================================================================
# Option 1: Use implementation WITHOUT LCEL
# ========================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without using LCEL. Langchain Execution Language (LCEL)
    This is a more manual approach where we handle the retrieval and prompt formatting ourselves.
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a single string
    context = format_docs(docs)

    # Step 3: Create the prompt by filling in the template
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke the LLM with the formatted prompt
    response = llm.invoke(messages)

    # Step 5: Return the response content
    return response
    

# ========================================================================
# Option 2: Use implementation WITH LCEL (Better Approach)
# ========================================================================
def create_retrieval_chain_with_lcel():
    """
    Creates a retrieval chain using LCEL (Langchain Execution Language).
    Returns a Chain that can be invoked with {"question":""}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
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
            context = itemgetter("question") | retriever | format_docs
        )
        |prompt_template 
        | llm | StrOutputParser()
    )
    return retrieval_chain
    

if __name__ == "__main__":
    print("Retrieving relevant documents...")

    # Query
    query = "What is Pinecone in Machine Learning?"

    # ==========================================
    # Option 0: Raw invocation without RAG
    # ==========================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 0: Raw LLM Invocation without RAG")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer without RAG:")
    print(result_raw)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 1: Without LCEL (Manual Approach)")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer without LCEL:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 2: With LCEL (Better Approach)")
    print("="*70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer with LCEL:")
    print(result_with_lcel)