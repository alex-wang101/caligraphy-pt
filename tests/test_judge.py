from calligraphy.agent.judge import agreement, build_cases, parse_judgement


def test_parse_judgement_json_and_fallback():
    j = parse_judgement('```json\n{"recognized_as": "帝", "is_correct_character": true, "score": 85, "problems": "ok"}\n```')
    assert j == {"score": 0.85, "recognized_as": "帝", "is_correct_character": True, "problems": "ok", "parse_ok": True}
    assert parse_judgement("<think>hmm</think> score: 140")["score"] == 1.0  # clamped to 0-100
    assert not parse_judgement("no idea")["parse_ok"]


def test_cases_cover_every_variant_and_score_as_expected():
    cases = build_cases(("smh_1196", "sgt_6080"), runs=_NoRuns())  # the wrong character needs a second one
    assert {c["variant"] for c in cases if c["key"] == "smh_1196"} >= {"perfect", "wrong_character", "scrambled", "thin_brush"}
    assert all(c["source"] is None for c in cases)


def test_agreement_perfect_judge():
    items = [
        {"variant": "perfect", "kind": "correct", "character": "帝", "reward": r, "judge": {"score": r, "recognized_as": "帝", "is_correct_character": True, "parse_ok": True}}
        for r in (0.1, 0.2, 0.85, 0.9)
    ]
    st = agreement(items)
    assert st["spearman"] == 1.0 and st["auc_good_bad"] == 1.0 and st["mae"] == 0.0 and st["reads_correct"] == 1.0


class _NoRuns:
    def list(self):
        return []
