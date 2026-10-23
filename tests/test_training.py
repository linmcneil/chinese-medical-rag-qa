import ast
import inspect

import pytest

from ragqa.training import pad_batch, tokenize_answer


class FakeTokenizer:
    chat_template = "synthetic"
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        # Synthetic deterministic prefix lets us inspect labels without model dependencies.
        return [101, 102, 103]

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def test_training_prefix_appears_once_and_only_answer_eos_are_supervised():
    row = tokenize_answer(FakeTokenizer(), "问题", "答案", 100)
    assert row["input_ids"] == [101, 102, 103, ord("答"), ord("案"), 0]
    assert row["labels"] == [-100, -100, -100, ord("答"), ord("案"), 0]
    assert row["input_ids"].count(101) == 1


def test_long_answer_is_rejected_instead_of_supervising_prefix():
    with pytest.raises(ValueError, match="exceeds"):
        tokenize_answer(FakeTokenizer(), "问题", "答案", 4)


def test_real_eos_attended_when_pad_equals_eos():
    a = tokenize_answer(FakeTokenizer(), "问题", "甲", 100)
    b = tokenize_answer(FakeTokenizer(), "问题", "甲乙丙", 100)
    result = pad_batch([a, b], pad_id=0)
    assert result["attention_mask"][0] == [1, 1, 1, 1, 1, 0, 0]
    assert result["labels"][0][-3:] == [0, -100, -100]


def test_optional_training_dependencies_not_imported_at_module_level():
    import ragqa.training

    tree = ast.parse(inspect.getsource(ragqa.training))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(n, "module", "") in ("torch", "datasets", "transformers") for n in imports)
