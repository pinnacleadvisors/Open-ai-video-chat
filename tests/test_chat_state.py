from server.pipeline.llm import ChatState


def test_messages_starts_with_system_prompt():
    s = ChatState(system_prompt="be helpful")
    msgs = s.messages()
    assert msgs == [{"role": "system", "content": "be helpful"}]


def test_add_user_and_assistant_round_trip():
    s = ChatState(system_prompt="sys")
    s.add_user("hello")
    s.add_assistant("hi there")
    s.add_user("how are you?")
    s.add_assistant("good")
    msgs = s.messages()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    assert msgs[1]["content"] == "hello"


def test_blank_messages_ignored():
    s = ChatState(system_prompt="sys")
    s.add_user("   ")
    s.add_assistant("")
    s.add_assistant(None)  # type: ignore[arg-type]
    assert len(s.messages()) == 1


def test_history_truncated_to_max():
    s = ChatState(system_prompt="sys", max_history=4)
    for i in range(10):
        s.add_user(f"u{i}")
        s.add_assistant(f"a{i}")
    msgs = s.messages()
    # system + last 4
    assert len(msgs) == 5
    assert msgs[1]["content"].startswith("u")
    assert msgs[-1]["content"] == "a9"


def test_reset_clears_history():
    s = ChatState(system_prompt="sys")
    s.add_user("hi")
    s.reset()
    assert s.messages() == [{"role": "system", "content": "sys"}]
