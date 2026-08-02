from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterTokenizer:
    """映射结果确定的字符级 tokenizer。"""

    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")

        # 排序使字符与 token ID 的映射不受集合遍历顺序影响。
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        # 字符在词表中的下标就是它的 token ID。
        # 当前样本定长且没有 padding，Bigram 也没有 attention 层，
        # 因此这里只返回 token ID，不生成 attention_mask。
        char_to_id = {
            character: index
            for index, character in enumerate(self.characters)
        }

        try:
            return [char_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        contains_invalid_id = any(
            token_id < 0 or token_id >= self.vocab_size
            for token_id in token_ids
        )

        if contains_invalid_id:
            raise ValueError("Token ID is outside the vocabulary")

        return "".join(self.characters[token_id] for token_id in token_ids)

    def to_dict(self) -> dict[str, list[str]]:
        return {"characters": list(self.characters)}

    @classmethod
    def from_dict(cls, values: dict[str, list[str]]) -> "CharacterTokenizer":
        return cls(tuple(values["characters"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(values)
