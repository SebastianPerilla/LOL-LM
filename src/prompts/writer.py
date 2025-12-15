SYSTEM_PROMPT = "You are a satirical news generator. When given a topic, generate a funny headline followed by the article."


def get_writer_prompt() -> str:
    """
    System prompt used for the fine-tuned writer model.
    Matches the system prompt used during fine-tuning.
    """
    return SYSTEM_PROMPT

