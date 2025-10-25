from src.experiments import assign_arm, sample_size_two_props, summarize_churn


def test_assign_arm_is_deterministic_and_balanced():
    xs = [f"user{i}" for i in range(1000)]
    arms = [assign_arm(x, "exp_v1") for x in xs]
    assert assign_arm("user123", "exp_v1") == assign_arm("user123", "exp_v1")
    assert set(arms).issubset({0, 1})
    p = sum(arms) / len(arms)
    assert 0.43 <= p <= 0.57


def test_sample_size_math_matches_rule_of_thumb():
    n = sample_size_two_props(p0=0.20, p1=0.17, alpha=0.05, power=0.80)
    assert 2400 <= n <= 2900


def test_summarize_churn_reports_rates_and_ci():
    control = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    treatment = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    report = summarize_churn(control, treatment, alpha=0.05)
    for k in ["p_c", "p_t", "diff", "ci_low", "ci_high", "n_c", "n_t", "z", "p_value"]:
        assert k in report
    assert 0 <= report["p_t"] < report["p_c"] <= 1
    assert report["diff"] < 0
    assert report["ci_low"] < report["diff"] < report["ci_high"]
    assert 0 <= report["p_value"] <= 1
