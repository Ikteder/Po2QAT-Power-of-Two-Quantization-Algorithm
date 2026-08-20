from po2qat.cli import interactive_argv


def test_interactive_launcher_selects_cnn_and_strong_profile():
    answers = iter(["1", "3"])
    assert interactive_argv(input_fn=lambda _: next(answers)) == ["run", "--model", "cnn", "--profile", "strong"]


def test_interactive_launcher_accepts_names_and_quick_default():
    answers = iter(["llm", ""])
    assert interactive_argv(input_fn=lambda _: next(answers)) == ["run", "--model", "llm", "--profile", "quick"]
