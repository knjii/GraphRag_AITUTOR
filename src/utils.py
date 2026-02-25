import logging
from pathlib import Path

from ollama import Client
from settings import Settings


_ollama_client = None


def get_ollama_client(settings: Settings | None = None, base_url: str | None = None):
    if Client is None:
        raise RuntimeError("ollama python package is not available")
    global _ollama_client
    cfg = settings or Settings()
    host = base_url or cfg.ollama_base_url
    if _ollama_client is None or base_url:
        _ollama_client = Client(host=host)
    return _ollama_client


def chat_with_ollama(
    message: str = "default",
    img_path: str | None = None,
    turn_off: bool = False,
    model: str | None = None,
    options: dict | None = None,
    base_url: str | None = None,
    settings: Settings | None = None,
) -> str | None:
    """
    Минимальный клиент к Ollama, совместимый с chunker.py.
    Возвращает текст ответа или None при turn_off=True.
    """
    cfg = settings or Settings()
    client = get_ollama_client(settings=cfg, base_url=base_url)
    model = model or cfg.ollama_model

    if options is None:
        options = {
            "temperature": float(cfg.temperature),
            "top_k": int(cfg.ollama_top_k),
            "num_ctx": int(cfg.ollama_num_ctx),
            "num_gpu": int(cfg.ollama_num_gpu),
            "num_batch": int(cfg.ollama_num_batch),
            "num_predict": int(cfg.ollama_num_predict),
        }

    if turn_off:
        client.chat(model=model, messages=[], keep_alive=0)
        return None

    if img_path:
        if message == "default":
            message = (
                "Тебе предоставлено изображение. Нужно описать его так, чтобы я мог по описанию "
                "восстановить себе материал этой картинки. Важно не включать избыточные детали. "
                "Не употребляй фразы по типу \"на картинке представлено\""
            )
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": message, "images": [img_path]}],
            options=options,
        )
    else:
        if message == "default":
            raise ValueError("Промпт для работы с текстом не задан")
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": message}],
            options=options,
        )

    return response["message"]["content"]


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_directories(settings: Settings) -> None:
    """Create expected directories if they are missing."""
    for path in (settings.pdf_dir, settings.markdown_dir, settings.chroma_dir):
        Path(path).mkdir(parents=True, exist_ok=True)


def chroma_has_data(settings: Settings) -> bool:
    chroma_path = Path(settings.chroma_dir)
    if not chroma_path.exists():
        return False
    return any(chroma_path.glob("**/*"))
