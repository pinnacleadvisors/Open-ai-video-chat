from server.pipeline.stt import VADStateMachine


def feed(vad, probs):
    out = []
    for p in probs:
        out.append(vad.step(p))
    return out


def test_initial_state_not_speaking():
    vad = VADStateMachine(threshold=0.5, silence_ms=300)
    assert not vad.speaking


def test_two_voiced_frames_transition_to_speaking():
    vad = VADStateMachine(threshold=0.5, silence_ms=300)
    feed(vad, [0.9, 0.9])
    assert vad.speaking


def test_single_voiced_frame_does_not_latch():
    vad = VADStateMachine(threshold=0.5, silence_ms=300)
    feed(vad, [0.9, 0.1, 0.1])
    assert not vad.speaking


def test_endpoint_fires_after_silence_window():
    # silence_ms=90 => 3 frames at 30ms
    vad = VADStateMachine(threshold=0.5, silence_ms=90)
    feed(vad, [0.9, 0.9])
    # now in speaking state
    a = vad.step(0.0)
    b = vad.step(0.0)
    c = vad.step(0.0)
    # c should be the endpoint
    assert a == (False, False)
    assert b == (False, False)
    assert c == (False, True)
    assert not vad.speaking


def test_reset_returns_clean_state():
    vad = VADStateMachine(threshold=0.5, silence_ms=300)
    feed(vad, [0.9, 0.9, 0.1, 0.1])
    vad.reset()
    assert not vad.speaking
    assert vad.silence_streak == 0
    assert vad.voiced_streak == 0


def test_intra_utterance_silence_does_not_end():
    # one frame of silence shouldn't end an utterance with a 90ms window
    vad = VADStateMachine(threshold=0.5, silence_ms=90)
    feed(vad, [0.9, 0.9])
    r1 = vad.step(0.0)
    r2 = vad.step(0.9)
    assert r1 == (False, False)
    assert vad.speaking
    assert r2[0]  # voiced again
