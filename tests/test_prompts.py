"""Tests for the 2-stage checker prompt template."""

from mt_llm.prompts import get_checker_prompt


def test_checker_prompt_includes_source_and_mt():
    prompt = get_checker_prompt("Hello world", "Hallo Welt")
    assert "Hello world" in prompt
    assert "Hallo Welt" in prompt


def test_checker_prompt_with_terms_lists_them():
    prompt = get_checker_prompt(
        source_text="The neural network learns features.",
        mt_output="Das neuronale Netz lernt Merkmale.",
        terms={"neural network": "Neuronales Netz"},
    )
    # The terms section for the actual task (not the example) should appear
    # below the example block.
    task_section = prompt.split("Task:", 1)[1]
    assert "neural network" in task_section
    assert "Neuronales Netz" in task_section


def test_checker_prompt_terminology_off_omits_terms_section():
    prompt = get_checker_prompt(
        source_text="x",
        mt_output="y",
        terms={"a": "b"},
        terminology_mode="off",
    )
    assert "Required terms" not in prompt


def test_checker_prompt_includes_few_shot_example():
    prompt = get_checker_prompt("anything", "anything")
    assert "Verbrauchsmodell" in prompt
    assert "Example:" in prompt


def test_checker_prompt_terms_off_omits_required_terms_in_example_too():
    prompt = get_checker_prompt("x", "y", terms={"x": "y"}, terminology_mode="off")
    assert "Required terms" not in prompt
