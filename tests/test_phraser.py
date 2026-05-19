from server.pipeline.tts import Phraser


def feed_all(tokens):
    p = Phraser()
    out = []
    for t in tokens:
        out.extend(p.feed(t))
    tail = p.flush()
    if tail:
        out.append(tail)
    return out


def test_emits_after_sentence_terminator_past_min_chars():
    text = "Hello there, this is a complete sentence. "
    out = feed_all(list(text))
    assert out == ["Hello there, this is a complete sentence."]


def test_holds_short_fragments_until_threshold():
    p = Phraser()
    assert p.feed("Hi.") == []
    assert p.feed(" ok.") == []
    tail = p.flush()
    assert tail == "Hi. ok."


def test_force_cuts_past_max_chars():
    long_token = "x" * (Phraser.MAX_CHARS + 5)
    out = feed_all([long_token])
    assert len(out) == 1
    assert len(out[0]) >= Phraser.MAX_CHARS


def test_multiple_phrases_in_a_long_stream():
    text = (
        "This is sentence one. This is sentence two, "
        "and it continues a bit more; then more again. "
        "Final bit here!"
    )
    out = feed_all(list(text))
    assert len(out) >= 3
    assert "".join(out).replace(" ", "") == text.replace(" ", "")


def test_streaming_byte_by_byte_matches_batch():
    text = "Hello world, this is a streaming test. And another."
    streamed = feed_all(list(text))
    batched = feed_all([text])
    assert streamed == batched


def test_flush_empty_returns_none():
    p = Phraser()
    assert p.flush() is None
