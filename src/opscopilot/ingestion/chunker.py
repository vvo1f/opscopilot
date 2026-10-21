"""Split page text into overlapping chunks.

Why chunk: one vector per whole document blurs many topics together, and one
vector per sentence loses context. A few hundred characters is the usual middle.
Why overlap: a fact sitting on a chunk boundary would otherwise be cut in two.
The size and overlap are settings, because we tune them against retrieval
metrics in phase 6 instead of guessing.
"""
from dataclasses import dataclass

# Try the coarsest boundary first (paragraph), then line, sentence, word.
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    page: int
    index: int   # position within the document, starting at 0
    text: str


def _split(text: str, max_chars: int, seps: list[str]) -> list[str]:
    """Cut text into pieces of at most max_chars, using the coarsest separator that works."""
    if len(text) <= max_chars:
        return [text]
    if not seps:
        # nothing to split on (one huge token): cut blindly
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    sep, finer = seps[0], seps[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, max_chars, finer)
    pieces = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")  # keep the separator, don't alter the text
        pieces.extend(_split(piece, max_chars, finer) if len(piece) > max_chars else [piece])
    return pieces


def _tail(text: str, n: int) -> str:
    """Last n characters of text, starting at a word boundary."""
    if n <= 0:
        return ""
    if len(text) <= n:
        return text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_text(text: str, max_chars: int = 300, overlap: int = 60) -> list[str]:
    pieces = _split(text.strip(), max_chars, SEPARATORS)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_chars:
            if current.strip():  # a run of blank lines must not become a chunk
                chunks.append(current.strip())
            # the overlap must still leave room for the new piece
            room = max_chars - len(piece)
            current = _tail(current, min(overlap, max(room, 0))) + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_pages(pages, max_chars: int = 300, overlap: int = 60) -> list[Chunk]:
    """Chunk each page separately, so every chunk has exactly one page number to cite."""
    chunks = []
    for page in pages:
        for text in chunk_text(page.text, max_chars, overlap):
            chunks.append(Chunk(page=page.number, index=len(chunks), text=text))
    return chunks


def document_header(pages, max_lines: int = 2, max_chars: int = 160) -> str:
    """The first non-empty lines of the document (title, parties, invoice number...)."""
    lines = [ln.strip() for ln in pages[0].text.splitlines() if ln.strip()]
    return " | ".join(lines[:max_lines])[:max_chars]


def add_headers(chunks: list[Chunk], header: str) -> list[Chunk]:
    """Prefix every chunk that doesn't already start with the document header.

    A dense embedding of "Total: 1.234,56 EUR" says nothing about which invoice
    it is. Repeating the title in each chunk fixes that, at the cost of a few
    extra characters per chunk. The setting chunk_header turns it off, so phase 6
    can measure whether it helps.
    """
    first_line = header.split(" | ")[0]
    for chunk in chunks:
        if not chunk.text.startswith(first_line):
            chunk.text = f"{header}\n{chunk.text}"
    return chunks
