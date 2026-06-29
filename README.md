# install Ollama (manual step)
https://ollama.com/download




# 🧠 LLM Agent with LangChain + Ollama

This project demonstrates how to build a **custom AI agent from scratch** using:

- LangChain (agent framework)
- Ollama (local LLM runtime)
- SQLite-based search tool (FAQ retrieval)
- Manual + agent-style tool calling

The goal is to understand how **agents, tools, memory, and loops** work under the hood.


# pull models
ollama pull qwen2.5:3b
ollama pull ollama pull qwen2.5:3b-instruct

# run ollama
- open power shell & write: where ollam (e.g. C:\Users\saba.yahyaa\AppData\Local\Programs\Ollama)
- open cmd command & write (add to the Path): set PATH=%PATH%;C:\Users\saba.yahyaa\AppData\Local\Programs\Ollama
-  open power shell & write: ollama serve
-  open power shell & write: ollama list (to get the loaded models)
---


# install postgres
1. download from:
2. install after selecting version 17, setting password (this is admin password)
3. open powershell and test: psql -U postgres,
   The psql shell is opend
4. open powershell and add to PATH: 
    [Environment]::SetEnvironmentVariable(
        "Path",
        $env:Path + ";C:\Program Files\PostgreSQL\17\bin",
        "User"
    )
    (OR) You can use cmd to add to PATH: setx PATH "%PATH%;C:\Program Files\PostgreSQL\17\bin"
5. in powershell test: psql --version (you need to get the version, psql (PostgreSQL) 17.10)
6. To open psql shell, open a powershell and write: psql -U postgres
7. Create an account: CREATE USER user_name WITH PASSWORD 'password'; 
   Note: the user_name is without '', but the password it has contain ''
8. check if creating a user is correct after openning the psql powershell:  \du,
   You need to have 2 users, the admin and the created one.
9. pgvector is NOT bundled with PostgreSQL since PostgreSQL only supports extensions, but does not include pgvector by default. You must install it separately
9.1 download: https://visualstudio.microsoft.com/visual-cpp-build-tools/ and complete the installation

Recommended install using docker that is installed in Ubutu under window
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=saba \
    -e POSTGRES_PASSWORD=saba \
    -e POSTGRES_DB=faq_postgres \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17

Then be sure that the installation is correct: docker pa, you need to get pgvector container, if not do: docker start pgvector


# 🚀 Features

- Local LLM using Ollama (no API costs)
- Tool-based reasoning (search function)
- Custom agent loop (manual + LangChain versions)
- Chat memory support
- Step-by-step agent execution debugging

---