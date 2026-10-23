"""Optional adapters using local random tiny models: wiring tests, not quality benchmarks."""

import pytest


def _tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    base = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "assistant": 3,
                "user": 4,
                "system": 5,
                "answer": 6,
                "question": 7,
            },
            unk_token="[UNK]",
        )
    )
    base.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]", model_max_length=256
    )
    tok.chat_template = (
        "{% for message in messages %}{{message['role']}} {{message['content']}} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    tok.save_pretrained(path)
    return tok


@pytest.mark.integration
def test_transformers_generator_loads_real_local_model_and_reuses_it(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers import GPT2Config, GPT2LMHeadModel
    from ragqa.generation import TransformersGenerator
    from ragqa.types import BackendError, Evidence

    path = tmp_path / "tiny-gpt"
    path.mkdir()
    _tokenizer(path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(path)
    generator = TransformersGenerator(str(path), max_new_tokens=3, max_input_tokens=256)
    evidence = [Evidence("E1", "c1", "d1", "question", "answer", "synthetic", 1.0)]
    assert isinstance(generator.generate("question", evidence), str)
    loaded = generator.model
    generator.generate("question", evidence)
    assert generator.model is loaded
    generator.max_input_tokens = 1
    with pytest.raises(BackendError):
        generator.generate("question", evidence)


@pytest.mark.integration
def test_cross_encoder_actual_api_and_model_cache(documents, tmp_path):
    pytest.importorskip("sentence_transformers")
    from transformers import BertConfig, BertForSequenceClassification
    from ragqa.chunking import chunk_documents
    from ragqa.rerank import CrossEncoderReranker
    from ragqa.types import Hit

    path = tmp_path / "tiny-bert"
    path.mkdir()
    _tokenizer(path)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=256,
            num_labels=1,
        )
    )
    model.save_pretrained(path)
    ranker = CrossEncoderReranker(str(path))
    hits = [Hit(c, 0.8) for c in chunk_documents(documents)]
    ranked = ranker.rerank("question", hits)
    assert {h.chunk.id for h in ranked} == {h.chunk.id for h in hits}
    assert all(h.rerank_score is not None for h in ranked)
    loaded = ranker.model
    ranker.rerank("question", hits)
    assert ranker.model is loaded


@pytest.mark.integration
def test_real_gradio_ui_builds_and_uses_pipeline(documents):
    pytest.importorskip("gradio")
    from ragqa.chunking import chunk_documents
    from ragqa.generation import ExtractiveGenerator
    from ragqa.pipeline import RAGPipeline
    from ragqa.retrieval import LexicalRetriever
    from ragqa.ui import build_ui

    pipeline = RAGPipeline(LexicalRetriever(chunk_documents(documents)), ExtractiveGenerator())
    ui = build_ui(pipeline)
    callbacks = list(ui.fns.values())
    text, detail = callbacks[0].fn("ALPHA-01")
    assert text == detail["text"] and detail["status"] == "answered"
    assert detail["evidence"][0]["source"] == "synthetic:a"
    ui.close()
