"""Text utilities for chunking and splitting text safely."""
import re
from typing import List


def combine_chunks(chunks: List[str], chunk_max: int) -> List[str]:
    """Combine small chunks into larger ones up to chunk_max characters."""
    combined_chunks = []
    current_chunk = ""
    for piece in chunks:
        # Try to add this piece to current_chunk
        if len(current_chunk) + len(piece) + (1 if current_chunk else 0) <= chunk_max:
            current_chunk += ("\n\n" if current_chunk else "") + piece
        else:
            if current_chunk:
                combined_chunks.append(current_chunk)
            current_chunk = piece

    if current_chunk:
        combined_chunks.append(current_chunk)

    return combined_chunks


def chop_text(text: str, n: int) -> list[str]:
    """Split text into chunks no longer than n, preserving sentence-ish breaks."""
    # Split text by sentence-ish boundaries, but keep delimiters
    # Match a period, semicolon, or double newline as separators
    pieces = re.split(r'(\.|;|\n\n)', text)

    # Reattach the separator (so "This is a sentence." instead of "This is a sentence" + ".")
    combined_pieces = []
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        if i + 1 < len(pieces) and pieces[i + 1] in {'.', ';', '\n\n'}:
            piece += pieces[i + 1]
            i += 2
        else:
            i += 1
        if piece.strip():
            combined_pieces.append(piece.strip())

    chunks = []
    current_chunk = ""

    for piece in combined_pieces:
        if len(current_chunk) + len(piece) + 1 <= n:
            current_chunk += (" " if current_chunk else "") + piece
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # now decide what to do with the big piece
            if len(piece) > n:
                # Hard split the big piece into multiple n-sized chunks
                for j in range(0, len(piece), n):
                    chunks.append(piece[j:j + n].strip())
                current_chunk = ""
            else:
                current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
