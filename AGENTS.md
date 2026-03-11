# AGENTS.md — PersonalMemoryBot

Guidance for agentic coding assistants working in this repository.

---

## Project Overview

PersonalMemoryBot is a single-file Chainlit application (`PersonalMemoryBot.py`)
that allows users to upload a PDF, embed it with Azure OpenAI embeddings (via
FAISS), and chat with it using a LangChain 1.x conversational agent backed by
Azure OpenAI (GPT-3.5-turbo). Conversation history is managed automatically by
a LangGraph `InMemorySaver` checkpointer.

**Stack:** Python · Chainlit · LangChain 1.x · langchain-openai · langchain-community ·
langchain-classic · LangGraph · Azure OpenAI · FAISS · pypdf · tiktoken

---

## Repository Layout

```
PersonalMemoryBot/
├── PersonalMemoryBot.py   # Entire application (single source file)
├── requirements.txt       # Runtime dependencies (partially pinned)
├── README.md
├── LICENSE
└── .gitignore
```

There are no sub-packages, modules, test directories, or build configuration
files beyond `requirements.txt`.

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Credentials — set as environment variables or in a `.env` file at the project
root (excluded from version control):

```bash
OPENAI_API_KEY="..."
OPENAI_API_BASE="https://<resource>.openai.azure.com/"
OPENAI_API_VERSION="2023-03-15-preview"
OPENAI_API_TYPE="azure"  # .env file: copy these to a .env file
```

---

## Running the Application

```bash
chainlit run PersonalMemoryBot.py
```

Opens the chat UI at `http://localhost:8000` by default.

---

## Build / Lint / Test Commands

This project has **no build system, linter configuration, or test suite.**

| Task | Command | Notes |
|---|---|---|
| Run app | `chainlit run PersonalMemoryBot.py` | localhost:8000 |
| Install deps | `pip install -r requirements.txt` | |
| Lint (ad hoc) | `flake8 PersonalMemoryBot.py` | no config file |
| Type check (ad hoc) | `mypy PersonalMemoryBot.py` | no config file |

**There are no tests.** If tests are added, use pytest:

```bash
pytest                                             # all tests
pytest tests/test_foo.py                           # single file
pytest tests/test_foo.py::test_function_name       # single function
pytest -s tests/test_foo.py::test_function_name    # with stdout
```

---

## Chainlit Patterns

- **`@cl.on_chat_start`** — runs once per session. Prompt for upload with
  `cl.AskFileMessage`, build the index and agent, store in `cl.user_session`
  (per-session key/value store).
- **`@cl.on_message`** — runs on every user message. Retrieve state via
  `cl.user_session.get(...)`.
- **`async with cl.Step(name="...")`** — shows progress in the UI for
  long-running operations (indexing, LLM calls). Wrap sync callables with
  `cl.make_async(fn)(...)`. All lifecycle handlers must be `async def` and
  `await cl.Message(...).send()`.

---

## Code Style Guidelines

### General

- Follow **PEP 8**. Use **4-space indentation**, no tabs.
- Keep lines under 100 characters where practical.
- Two blank lines between top-level definitions (PEP 8).
- Use `# Comment` above logical blocks; avoid end-of-line comments for
  anything more than a very short note.

### Imports

Group in this order, separated by a blank line:

1. Standard library (`import re`, `from typing import List`, …)
2. Third-party packages (`import chainlit as cl`, `from langchain_core …`, …)

Sort alphabetically within each group. Do **not** introduce unused imports.

### Naming Conventions

| Construct | Convention | Example |
|---|---|---|
| Functions / handlers | `snake_case` | `parse_pdf`, `build_agent`, `on_message` |
| Variables / parameters | `snake_case` | `doc_chunks`, `name_of_file`, `pages` |
| Constants (inline) | literal values | `chunk_size=2000`, `chunk_overlap=0` |
| File names | `PascalCase` (existing) | `PersonalMemoryBot.py` |

No classes exist in this project. If added, use `PascalCase`.

### Type Hints

- Annotate all function signatures (parameters and return type).
- Use the `typing` module style (`List[str]`) — the codebase targets Python
  3.8+ compatibility. Do **not** use built-in generics (`list[str]`) unless the
  minimum version is explicitly raised.
- Do not annotate local variables unless it genuinely aids readability.
- Never import from `typing` what is not used.

### Docstrings

- Required for non-trivial functions where the name alone is not self-documenting.
- Use a single-line summary sentence (`text_to_docs` is the canonical example).
- `@tool`-decorated functions **must** have a docstring — LangChain uses it as
  the tool description passed to the LLM.

### Error Handling

- Wrap every external API call (embeddings, FAISS, LLM) in `try/except` and
  surface errors via `await cl.Message(content=f"...{e}").send()` followed by
  `return`.
- Never use a bare `except:`; catch the most specific type available, falling
  back to `Exception` only when necessary.
- Raise `RuntimeError` with a descriptive message from helper functions;
  handle it in the Chainlit handler where user feedback can be sent.

---

## Import Paths (LangChain 1.x package split)

| Symbol | Package |
|---|---|
| `AzureChatOpenAI`, `AzureOpenAIEmbeddings` | `langchain_openai` |
| `FAISS` | `langchain_community.vectorstores` |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` |
| `Document` | `langchain_core.documents` |
| `ChatPromptTemplate` | `langchain_core.prompts` |
| `HumanMessage`, `AIMessage` | `langchain_core.messages` |
| `tool` decorator | `langchain_core.tools` |
| `create_agent` | `langchain.agents` |
| `create_retrieval_chain` | `langchain_classic.chains` |
| `create_stuff_documents_chain` | `langchain_classic.chains.combine_documents` |
| `InMemorySaver` | `langgraph.checkpoint.memory` |

Never use the old monolithic paths (`from langchain import LLMChain`,
`from langchain.chat_models import AzureChatOpenAI`, etc.) — removed in 1.x.

---

## Known Issues / Technical Debt

- `chainlit`, `langgraph`, `langchain-classic`, and `pypdf` are unpinned in
  `requirements.txt` — pin before deploying to production.
- `build_agent` return type is annotated as `object`; narrow to
  `CompiledStateGraph` from `langgraph.graph.state` when convenient.
  `MessagesPlaceholder` is imported but unused — remove if not needed.
- No tests, no CI/CD, no enforced linter or formatter.
