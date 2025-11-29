from langchain_ollama import ChatOllama

def getModel():
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )
    return llm