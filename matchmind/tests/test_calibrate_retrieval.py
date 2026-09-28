from matchmind.scripts.calibrate_retrieval import BAD_PROBES, choose_threshold


def test_threshold_sits_in_the_gap_between_bad_and_good():
    # Reranker confidences: good probes near 1.0, off-domain ones near 0.
    assert choose_threshold(good=[0.82, 0.99, 1.0], bad=[0.0, 0.01]) == 0.415


def test_threshold_clamped_to_sane_range():
    assert choose_threshold(good=[0.99], bad=[0.98]) <= 0.9
    assert choose_threshold(good=[0.05], bad=[0.0]) >= 0.1


def test_offside_rule_is_not_an_off_domain_probe():
    # It legitimately retrieves "Offside Trap"; as a "bad" probe it forced the
    # threshold to 0.9 and made nearly every agent query re-retrieve.
    assert not any("offside" in p for p in BAD_PROBES)
