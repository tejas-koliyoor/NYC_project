import hashlib
import math
from typing import List, Dict


def assign_arm(entity_id: str, experiment_id: str) -> int:
    """Deterministic 50/50 assignment. Return 0 (control) or 1 (treatment)."""
    key = f"{entity_id}|{experiment_id}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    # take a small hex slice -> int -> mod 2
    return int(h[-2:], 16) % 2


def sample_size_two_props(
    p0: float, p1: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Approx per-arm n for detecting |p1-p0| at alpha/power. Return ceil int."""
    delta = abs(p1 - p0)
    if delta == 0:
        raise ValueError("Effect size (|p1 - p0|) must be > 0")

    pbar = 0.5 * (p0 + p1)

    # two-sided alpha: use z for alpha/2
    # common engineering constants for quick calcs:
    z_alpha = 1.96 if abs(alpha - 0.05) < 1e-9 else abs(
        math.sqrt(2) * math.erfcinv(alpha)
    )
    # 80% power -> z_beta ~ 0.84
    z_beta = 0.84 if abs(power - 0.80) < 1e-9 else abs(
        math.sqrt(2) * math.erfcinv(2 * (1 - power))
    )

    term1 = z_alpha * math.sqrt(2 * pbar * (1 - pbar))
    term2 = z_beta * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    n = ((term1 + term2) ** 2) / (delta ** 2)
    return math.ceil(n)


def summarize_churn(
    control: List[int], treatment: List[int], alpha: float = 0.05
) -> Dict[str, float]:
    """Return dict with p_c, p_t, diff, ci_low, ci_high, n_c, n_t, z, p_value."""
    n_c, n_t = len(control), len(treatment)
    if n_c == 0 or n_t == 0:
        raise ValueError("Both arms must be non-empty")

    p_c = sum(control) / n_c
    p_t = sum(treatment) / n_t
    diff = p_t - p_c

    se = math.sqrt(
        (p_c * (1 - p_c)) / n_c + (p_t * (1 - p_t)) / n_t
    )

    # standard normal CDF via erf
    phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    # z critical for two-sided alpha
    zcrit = 1.96 if abs(alpha - 0.05) < 1e-9 else abs(
        math.sqrt(2) * math.erfcinv(alpha)
    )

    if se == 0:
        z = 0.0
        p_value = 1.0
        ci_low, ci_high = diff, diff
    else:
        z = diff / se
        p_value = 2.0 * (1.0 - phi(abs(z)))
        ci_low = diff - zcrit * se
        ci_high = diff + zcrit * se

    return {
        "p_c": p_c,
        "p_t": p_t,
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_c": n_c,
        "n_t": n_t,
        "z": z,
        "p_value": p_value,
    }
