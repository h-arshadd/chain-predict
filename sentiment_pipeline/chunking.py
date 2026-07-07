"""
chunking.py
-----------
Shared helper: every HuggingFace model has its own max token limit, and
its own tokenizer (different models split text into tokens differently -
there's no single "chunk size" that works for all of them). This is the
one place that logic lives, so every model file just calls this instead
of re-implementing chunking each time.
"""


def split_into_token_chunks(tokenizer, text: str, max_tokens: int) -> list:
    """
    Returns a list of decoded text chunks, each within max_tokens according
    to the given tokenizer. A short post just comes back as a single-item
    list, so callers can always loop over the result the same way whether
    chunking actually happened or not.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return [""]

    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in token_chunks]