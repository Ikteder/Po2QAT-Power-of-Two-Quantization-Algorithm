from po2qat.cli import expand_model_selection, interactive_argv


def test_interactive_launcher_selects_cnn_and_strong_profile():
    answers = iter(["1", "3"])
    assert interactive_argv(input_fn=lambda _: next(answers)) == ["run", "--model", "cnn", "--profile", "strong"]


def test_interactive_launcher_accepts_names_and_quick_default():
    answers = iter(["resnet", ""])
    assert interactive_argv(input_fn=lambda _: next(answers)) == ["run", "--model", "resnet50", "--profile", "quick"]


def test_interactive_launcher_accepts_llm_choice():
    answers = iter(["5", "2"])
    assert interactive_argv(input_fn=lambda _: next(answers)) == ["run", "--model", "llm", "--profile", "quick"]


def test_interactive_launcher_accepts_vgg19_choice():
    answers = iter(["3", "1"])
    assert interactive_argv(input_fn=lambda _: next(answers)) == ["run", "--model", "vgg19", "--profile", "smoke"]


def test_model_groups_keep_large_models_opt_in():
    assert expand_model_selection("classroom") == ["cnn", "vit", "llm"]
    assert expand_model_selection("all") == ["cnn", "vit", "vgg19", "resnet50", "llm"]
