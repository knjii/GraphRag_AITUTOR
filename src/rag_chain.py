try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:  # pragma: no cover - fallback for older langchain stacks
    from langchain_classic.chains import create_retrieval_chain  # type: ignore
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from llm import get_chat_model
from retriever import build_retriever
from settings import Settings
from utils import get_logger

logger = get_logger("rag_chain")


def build_rag_chain(settings: Settings) -> Runnable:
    """LCEL RAG: retrieve by user input, then answer with optional chat history."""
    retriever = build_retriever(settings)
    llm = get_chat_model(settings)

    qa_system_prompt = settings.qa_system_prompt
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain
