import asyncio
import uuid
import os
import faiss
import git
from datetime import datetime
# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import getValueFromConfig

class _KnowledgeBaseHandler:
    """Internal memory store (not exposed directly)"""
    
    def __init__(self):
        self._knowledge_store = None
        self._initialized = False
        self._persist_dir = "./faiss_knowledge_base"
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the memory store"""
        async with self._lock:
            if not self._initialized:
                memory_embedding_model = getValueFromConfig("knowledge_base", "embedding_model")
                embeddings = HuggingFaceEmbeddings(model_name=memory_embedding_model)

                # Create persist directory if it doesn't exist
                os.makedirs(self._persist_dir, exist_ok=True)

                # Try to load existing FAISS index
                index_path = os.path.join(self._persist_dir, "index")
                if os.path.exists(f"{index_path}.faiss"):
                    print("Loading existing FAISS index...")
                    self._knowledge_store = FAISS.load_local(
                        self._persist_dir,
                        embeddings,
                        allow_dangerous_deserialization=True  # Required for pickle loading
                    )
                    print("FAISS knowledge base loaded successfully!")
                    # load_all_documents_from_version_control_history(self._knowledge_store)
                    # self._save_index()
                else:
                    print("Creating new FAISS index...")
                    embedding_dim = len(embeddings.embed_query("hello world"))
                    index = faiss.IndexFlatL2(embedding_dim)

                    self._knowledge_store = FAISS(
                        embedding_function=embeddings,
                        index=index,
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={},
                    )
                    print("FAISS empty knowledge base created successfully!")
                    load_all_documents_from_version_control_history(self._knowledge_store)
                    self._save_index()

                self._initialized = True
                print("Knowledge base initialized successfully!")
    
    def _save_index(self):
        """Save FAISS index to disk"""
        if self._knowledge_store:
            self._knowledge_store.save_local(self._persist_dir)
    
    async def cleanup(self):
        """Cleanup the memory store"""
        async with self._lock:
            if self._initialized:
                self._save_index()
                self._knowledge_store = None
                self._initialized = False
                print("Memory store cleaned up successfully!")
    
    @property
    def knowledge_store(self):
        """Get the memory store instance"""
        return self._knowledge_store

# Module-level singleton instance
_manager = _KnowledgeBaseHandler()

# Public API
async def initialize_knowledge_base():
    await _manager.initialize()

async def cleanup_knowledge_base():
    await _manager.cleanup()

def get_knowledge_store():
    return _manager.knowledge_store

def _save_knowledge_store():
    """Helper to save the FAISS index"""
    _manager._save_index()


def load_all_documents_from_version_control_history(knowledge_store):

    """
    Load all git commits and their diffs from a repository into the knowledge base.
    """
    repo_path = "/Users/dhrsharm/Documents/projects/delrina/web-designer-components"
    
    try:
        # Open the git repository
        repo = git.Repo(repo_path)
        
        # Get all commits
        commits = list(repo.iter_commits('dev', max_count=100))
        print(f"Found {len(commits)} commits to process...")

        total_documents = 0
        batch_size = 10
        for i in range(0, len(commits), batch_size):
            batch_commits = commits[i:i + batch_size]
        
            documents = []
            
            for j, commit in enumerate(batch_commits):
                documents.extend(get_documents_per_file_from_commit(commit))
                
                # Progress indicator
                if (j + 1) % 5 == 0:
                    print(f"Processed {i + j + 1}/{len(commits)} commits...")
            
            # Add all documents to the knowledge store
            if documents:
                print(f"Adding {len(documents)} documents to knowledge base...(commits {i+1}-{min(i+batch_size, len(commits))})")
                knowledge_store.add_documents(documents)
                total_documents += len(documents)
                print("Successfully loaded batch of git history into knowledge base!")
            else:
                print("No documents to add.")
            
        print(f"Successfully loaded {total_documents} documents from git history!")
    except git.exc.InvalidGitRepositoryError:
        print(f"Error: {repo_path} is not a valid git repository")
    except Exception as e:
        print(f"Error loading git history: {str(e)}")


def get_documents_per_file_from_commit(commit):
    # Initialize text splitter for large diffs
    # nomic-embed-text has ~2048 token limit, so we use smaller chunks to be safe
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # characters, not tokens (roughly ~250 tokens)
        chunk_overlap=100,  # some overlap for context
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    # Extract commit information
    commit_sha = commit.hexsha
    commit_author = commit.author.name
    commit_email = commit.author.email
    commit_date = datetime.fromtimestamp(commit.committed_date).isoformat()
    commit_message = commit.message.strip()
    
    # Get the diff for this commit
    # Compare with parent (if exists)
    if commit.parents:
        parent = commit.parents[0]
        diffs = parent.diff(commit, create_patch=True)
    else:
        # First commit - show all files as added
        diffs = commit.diff(git.NULL_TREE, create_patch=True)
    
    # Process each file diff
    for diff in diffs:
        # Get file path
        file_path = diff.b_path if diff.b_path else diff.a_path
        
        # Get change type
        change_type = "modified"
        if diff.new_file:
            change_type = "added"
        elif diff.deleted_file:
            change_type = "deleted"
        elif diff.renamed_file:
            change_type = "renamed"
        
        # Get the actual diff text
        diff_text = ""
        if diff.diff:
            try:
                diff_text = diff.diff.decode('utf-8')
            except UnicodeDecodeError:
                diff_text = "[Binary file or encoding issue]"
        
        # Skip if diff is empty or binary
        if not diff_text or diff_text == "[Binary file or encoding issue]":
            continue
            
        # Truncate very large diffs
        max_diff_length = 10000  # characters
        if len(diff_text) > max_diff_length:
            diff_text = diff_text[:max_diff_length] + "\n\n[... diff truncated due to length ...]"
        
        # Create the base content
        base_content = f"""Commit: {commit_sha[:8]}
Author: {commit_author} <{commit_email}>
Date: {commit_date}
Message: {commit_message}

File: {file_path}
Change Type: {change_type}
"""
        
        # Chunk the diff if it's still large
        if len(diff_text) > 800:
            # Split diff into chunks
            diff_chunks = text_splitter.split_text(diff_text)
            
            # Create a document for each chunk
            for chunk_idx, chunk in enumerate(diff_chunks):
                page_content = f"""{base_content}
Diff (Part {chunk_idx + 1}/{len(diff_chunks)}):
{chunk}
"""
                doc = Document(
                    page_content=page_content.strip(),
                    metadata={
                        "source": "git_commit",
                        "commit_sha": commit_sha,
                        "commit_sha_short": commit_sha[:8],
                        "author": commit_author,
                        "author_email": commit_email,
                        "date": commit_date,
                        "message": commit_message,
                        "file_path": file_path,
                        "change_type": change_type,
                        "chunk": chunk_idx,
                        "total_chunks": len(diff_chunks)
                    }
                )
                documents.append(doc)
        else:
            # Small diff, no chunking needed
            page_content = f"""{base_content}
Diff:
{diff_text}
"""
            doc = Document(
                page_content=page_content.strip(),
                metadata={
                    "source": "git_commit",
                    "commit_sha": commit_sha,
                    "commit_sha_short": commit_sha[:8],
                    "author": commit_author,
                    "author_email": commit_email,
                    "date": commit_date,
                    "message": commit_message,
                    "file_path": file_path,
                    "change_type": change_type,
                }
            )
        documents.append(doc)
    
    return documents

def search_git_commits_based_on_query(query: str, limit: int = 20):
    """
    Search through git commits and diffs using semantic search.
    
    Args:
        query: The search query (e.g., "authentication changes", "bug fixes in login")
        limit: Maximum number of results to return
    
    Returns:
        List of relevant documents with their similarity scores
    """
    store = get_knowledge_store()
    if not store:
        return []
    
    # Search with similarity scores
    results = store.similarity_search_with_score(
        query=query,
        k=limit,
        filter={"source": "git_commit"}  # Only search git commits
    )
    
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in results
    ]

# def getMemoriesForUserBasedOnQuery(user_id: str, query: str, limit: int = 3):
#     """Search memories using FAISS similarity search"""
#     store = get_memory_store()
#     if not store:
#         return []
    
#     # Search with more results to filter by user
#     results = store.similarity_search_with_score(
#         query=query,
#         k=limit * 10,  # Fetch more to filter by user_id
#         filter={"user_id": user_id, "namespace": "memories"}
#     )
    
#     # Filter and limit results
#     filtered_results = [
#         doc.page_content for doc, score in results
#         if doc.metadata.get("user_id") == user_id and doc.metadata.get("namespace") == "memories"
#     ][:limit]
    
#     return filtered_results

# async def updateMemoryForUser(user_id: str, messages, memory_creator):
#     # Namespace the memory
#     namespace = (user_id, "memories")

#     # Create a new memory ID
#     memory_id = str(uuid.uuid4())

#     # Creating the memory create prompt
#     memory_create_prompt_template = PromptTemplate.from_template(MEMORY_CREATE_PROMPT)
#     conversation_messages = "\n".join(["user: " + msg.content if isinstance(msg, HumanMessage) else "assistant: " + msg.content for msg in messages])
    
#     memory_create_prompt = memory_create_prompt_template.invoke({"conversation_messages": conversation_messages})
    
#     memory_prompt_template = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful assistant designed to create concise summaries (short-term memories) of conversations."),
#         ("human", "{input}")
#     ])
#     prompt = memory_prompt_template.format_messages(input=memory_create_prompt.to_string())
#     # Creating the memory
#     stringified_memory = await memory_creator(prompt)

#     doc = Document(
#         page_content=stringified_memory,
#         metadata={
#             "user_id": user_id,
#             "namespace": "memories",
#             "memory_id": memory_id
#         }
#     )
#     get_memory_store().add_documents([doc])
    
#     # Save to disk after adding
#     _save_memory_store()
#     print(f"Memory saved and persisted to disk: \n\n {stringified_memory} \n\n")