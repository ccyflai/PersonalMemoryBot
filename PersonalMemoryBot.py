# Import necessary modules
import re
from typing import List

import chainlit as cl
from langchain.agents import create_agent
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from pypdf import PdfReader


# Parse a PDF file and extract its text content per page
def parse_pdf(path: str) -> List[str]:
    pdf = PdfReader(path)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Convert a list of page strings to a list of Documents with metadata
def text_to_docs(text: List[str]) -> List[Document]:
    """Converts a list of page strings to a list of Documents with metadata."""
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={"page": doc.metadata["page"], "chunk": i},
            )
            # Add source as metadata
            chunk_doc.metadata["source"] = (
                f"{chunk_doc.metadata['page']}-{chunk_doc.metadata['chunk']}"
            )
            doc_chunks.append(chunk_doc)
    return doc_chunks


# Build a FAISS vector index from document chunks
async def build_index(pages: List[Document]) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        chunk_size=1,
    )
    try:
        index = FAISS.from_documents(pages, embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to build index: {e}") from e
    return index


# Build a conversational agent that can answer questions from the PDF index
def build_agent(index: FAISS) -> object:
    llm = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    # Build the RAG chain used as the agent's PDF tool
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question using only the following context from the PDF:\n\n{context}",
        ),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(index.as_retriever(), combine_docs_chain)

    # Wrap the retrieval chain as a tool callable by the agent
    @tool
    def pdf_qa(question: str) -> str:
        """Answer questions about the uploaded PDF document."""
        result = retrieval_chain.invoke({"input": question})
        return result.get("answer", "No answer found.")

    system_prompt = (
        "You are a helpful assistant that answers questions about an uploaded PDF document. "
        "Use the pdf_qa tool to look up information from the document. "
        "Always ground your answers in what the document says."
    )

    # create_agent (LangChain 1.x) returns a compiled LangGraph agent.
    # Conversation history is managed automatically via the InMemorySaver checkpointer.
    agent = create_agent(
        model=llm,
        tools=[pdf_qa],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    return agent


@cl.on_chat_start
async def on_chat_start() -> None:
    """Prompt the user to upload a PDF, then build the index and agent."""
    files = await cl.AskFileMessage(
        content="Welcome to **PersonalMemoryBot**! Upload a PDF to get started.",
        accept=["application/pdf"],
        max_size_mb=20,
    ).send()

    pdf_file = files[0]
    name_of_file = pdf_file.name

    await cl.Message(content=f"Processing `{name_of_file}`...").send()

    # Parse and chunk the PDF
    try:
        raw_pages = parse_pdf(pdf_file.path)
        pages = text_to_docs(raw_pages)
    except Exception as e:
        await cl.Message(content=f"Failed to parse PDF: {e}").send()
        return

    if not pages:
        await cl.Message(content="Could not extract any text from the PDF.").send()
        return

    # Build the FAISS index
    async with cl.Step(name="Indexing document") as step:
        try:
            index = await build_index(pages)
            step.output = "Embeddings built successfully."
        except RuntimeError as e:
            await cl.Message(content=str(e)).send()
            return

    # Build the agent (includes its own InMemorySaver for conversation history)
    try:
        agent = build_agent(index)
    except Exception as e:
        await cl.Message(content=f"Failed to build agent: {e}").send()
        return

    # Persist agent and a fixed thread_id for the session
    cl.user_session.set("agent", agent)
    cl.user_session.set("thread_id", "session")
    cl.user_session.set("file_name", name_of_file)

    await cl.Message(
        content=f"Ready! Ask me anything about `{name_of_file}`."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle each user message by running the conversational agent."""
    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")

    if agent is None:
        await cl.Message(
            content="No document loaded. Please restart the chat and upload a PDF."
        ).send()
        return

    config = {"configurable": {"thread_id": thread_id}}

    async with cl.Step(name="Thinking"):
        try:
            result = await cl.make_async(agent.invoke)(
                {"messages": [HumanMessage(content=message.content)]},
                config=config,
            )
            # The agent returns a state dict; the last AIMessage is the response
            ai_messages = [
                m for m in result.get("messages", []) if isinstance(m, AIMessage)
            ]
            res = ai_messages[-1].content if ai_messages else "No response generated."
        except Exception as e:
            await cl.Message(content=f"Error generating response: {e}").send()
            return

    await cl.Message(content=res).send()
