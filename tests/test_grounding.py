import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "judges"))

import build_set  # noqa: E402
import grounding  # noqa: E402


def test_pure_questions_are_questions():
    assert grounding.is_question("And then she shrinks?")
    assert grounding.is_question("Ha! And then she shrinks?")
    assert grounding.is_question("So he's against government entirely?")


def test_statement_then_question_is_a_claim():
    assert not grounding.is_question("But the stink came before the sewers. What happened in 1858?")
    assert not grounding.is_question("No. He will cheerfully obey those who know better.")


def _line(text, first):
    return {"text": text, "src": ["S1"], "check": {"first": first, "local": None}}


def test_rates_leave_questions_out_of_claim_lines():
    lines = [_line("She shrinks to ten inches.", "supported"),
             _line("And then she shrinks?", "supported"),
             _line("What would grown-up Alice do?", "unsupported"),
             _line("The Hatter says it is always six o'clock.", "partly")]
    r = grounding.rates(lines)
    assert r["claim_lines"] == 2
    assert r["question_lines"] == 2
    assert r["questions_failed_first"] == 1
    assert r["not_supported_rate_first"] == 0.5


def test_number_perturbation_moves_a_year_by_ten():
    out, edit = build_set.perturb_number("In 1858 the Thames stank.", "The summer of 1858 was hot.")
    assert out == "In 1868 the Thames stank." and edit == "1858 -> 1868"


def test_number_perturbation_skips_figures_already_in_the_source():
    assert build_set.perturb_number("It cost 3 pounds.", "It cost 3 or 4 pounds.") is None


def test_name_swap_stays_within_kind():
    groups = {"person": ["Bazalgette", "Chadwick"], "place": ["Lambeth", "Chelsea"]}
    out, edit = build_set.perturb_name("Bazalgette built it.", "Bazalgette built the sewers.", groups, random.Random(0))
    assert out == "Chadwick built it."


def test_negation_avoids_double_negatives_and_handles_curly_apostrophes():
    assert build_set.perturb_negation("There is no appeal.", "") is None
    out, _ = build_set.perturb_negation("She didn’t see it.", "She saw nothing.")
    assert out == "She did see it."
    out, _ = build_set.perturb_negation("The river was foul.", "The river was foul that summer.")
    assert out == "The river was not foul."


def test_parse_json_takes_the_last_object_when_the_model_corrects_itself():
    import llm
    raw = ('{"verdicts":[{"n":1,"verdict":"supported"}]}\n\nCorrection to line 1: it says "x".\n\n'
           '{"verdicts":[{"n":1,"verdict":"partly"}]}\n```')
    assert llm.parse_json(raw) == {"verdicts": [{"n": 1, "verdict": "partly"}]}
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Here: [1, 2]') == [1, 2]
