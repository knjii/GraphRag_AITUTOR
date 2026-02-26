try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:  # pragma: no cover - fallback for older langchain stacks
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain  # type: ignore
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from llm import get_chat_model
from retriever import build_retriever
from settings import Settings
from utils import get_logger

logger = get_logger("rag_chain")


def build_rag_chain(settings: Settings) -> Runnable:
    """LCEL two-step RAG: rewrite question with history, then retrieve+answer."""
    retriever = build_retriever(settings)
    llm = get_chat_model(settings)

    # contextualize_q_system_prompt = (
    #     "Given a chat history and the latest user question "
    #     "which might reference context in the chat history, "
    #     "formulate a standalone question which can be understood "
    #     "without the chat history. Do NOT answer the question, "
    #     "just reformulate it if needed and otherwise return it as is."
    # )
    contextualize_q_system_prompt = settings.contextualize_q_system_prompt

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = settings.qa_system_prompt
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
