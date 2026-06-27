"""
Text → uint16 token ID batched encoder for the data pipeline.

Takes Document objects from a source stream, encodes them with the
loaded BPE tokenizer, and prepends <|endoftext|> (ID=0) as a document
separator — this is how GPT-style models mark document boundaries in
the token stream.

Token stream layout:
    <|endoftext|> doc1_tokens... <|endoftext|> doc2_tokens... ...

The leading <|endoftext|> per document lets the model learn to predict
the first real token from context that ends with a separator.

Usage:
    from tokenizer.serialization.load import load_tokenizer
    from data.sources.fineweb import FineWebEduStream
    from data.preprocessing.tokenize_dataset import DocumentTokenizer

    tok = load_tokenizer("tokenizer/saved/tokenizer.json")
    stream = FineWebEduStream(token_budget=5_000_000_000)
    tokenizer = DocumentTokenizer(tok)

    for token_ids in tokenizer.encode_stream(stream):
        # token_ids: list[int] — one document's tokens incl. leading EOT
        shard_writer.add(token_ids)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from tqdm import tqdm

if TYPE_CHECKING:
    from data.sources.fineweb import Document
    from tokenizer.serialization.load import Tokenizer

# Document separator — must match SPECIAL_TOKENS["<|endoftext|>"] = 0
_EOT_ID = 0


class DocumentTokenizer:
    """
    Encodes a stream of Document objects to uint16 token ID lists.

    Each document is encoded as:
        [<|endoftext|>] + encode(text)

    Args:
        tokenizer: Loaded Tokenizer from load_tokenizer().
        add_eot:   Prepend <|endoftext|> per document (default True).
                   Set False only for custom separator schemes.
    """

    def __init__(self, tokenizer: "Tokenizer", add_eot: bool = True) -> None:
        self.tokenizer = tokenizer
        self.add_eot   = add_eot
        self._eot_id   = tokenizer.special_tokens["<|endoftext|>"]

    def encode_document(self, text: str) -> list[int]:
        """
        Encode a single document text to a token ID list.

        Returns:
            [<|endoftext|>] + bpe_tokens  (uint16-safe ints)
        """
        ids = self.tokenizer.encode(text)
        if self.add_eot:
            return [self._eot_id] + ids
        return ids

    def encode_stream(
        self,
        documents: Iterator["Document"],
        log_every: int = 100_000,
        progress: bool = True,
    ) -> Iterator[list[int]]:
        """
        Encode a document stream, yielding token ID lists one doc at a time.

        Args:
            documents: Iterator of Document objects.
            log_every: Update postfix every N documents (used with progress).
            progress:  Show tqdm bar over documents (default True).

        Yields:
            list[int]: Token IDs for one document (incl. leading EOT).
        """
        total_tokens = 0
        total_docs   = 0

        doc_iter = documents
        if progress:
            doc_iter = tqdm(
                documents,
                desc="tokenize",
                unit="doc",
                mininterval=0.5,
                dynamic_ncols=True,
            )

        for doc in doc_iter:
            ids = self.encode_document(doc.text)
            total_tokens += len(ids)
            total_docs   += 1

            if progress and log_every and total_docs % log_every == 0:
                if isinstance(doc_iter, tqdm):
                    doc_iter.set_postfix(
                        tokens=f"{total_tokens / 1e6:.1f}M",
                        avg=f"{total_tokens / total_docs:.0f}/doc",
                    )

            yield ids

        if progress and isinstance(doc_iter, tqdm):
            doc_iter.close()

        print(
            f"  [tokenize] done: {total_docs:,} docs  "
            f"{total_tokens / 1e6:.1f}M tokens"
        )
