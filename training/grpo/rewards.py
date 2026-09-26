"""Deterministic, dependency-light reward components for WG GRPO rollouts."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from training.common import parse_top3


POSITION_WEIGHTS = (1.0, 0.5, 0.25)


def _completion_text(completion: Any) -> str:
    """Accept both TRL standard completions and conversational completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return ""


def _score_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for tariff_id, score in value.items():
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            result[str(tariff_id)] = float(score)
    return result


def _normalized_utilities(reward_by_tariff_id: dict[str, float]) -> dict[str, float]:
    if not reward_by_tariff_id:
        return {}
    values = list(reward_by_tariff_id.values())
    low, high = min(values), max(values)
    if high == low:
        return {tariff_id: 1.0 for tariff_id in reward_by_tariff_id}
    return {
        tariff_id: (value - low) / (high - low)
        for tariff_id, value in reward_by_tariff_id.items()
    }


def strict_json_reward(completions: Sequence[Any], **_: Any) -> list[float]:
    """One only for the exact required JSON schema; prose/fences receive zero."""
    return [float(parse_top3(_completion_text(completion))[1]) for completion in completions]


def valid_tariffs_reward(
    completions: Sequence[Any],
    reward_by_tariff_id: Sequence[dict[str, float]],
    **_: Any,
) -> list[float]:
    """Fraction of three positions occupied by distinct offered tariff IDs."""
    rewards: list[float] = []
    for completion, raw_scores in zip(completions, reward_by_tariff_id, strict=True):
        predicted, _ = parse_top3(_completion_text(completion))
        offered = set(_score_map(raw_scores))
        valid_unique = {tariff_id for tariff_id in predicted[:3] if tariff_id in offered}
        rewards.append(min(len(valid_unique), 3) / 3.0)
    return rewards


def calibrated_top1_reward(
    completions: Sequence[Any],
    reward_by_tariff_id: Sequence[dict[str, float]],
    **_: Any,
) -> list[float]:
    """Min-max-normalized frozen business reward of the generated first pick."""
    rewards: list[float] = []
    for completion, raw_scores in zip(completions, reward_by_tariff_id, strict=True):
        predicted, _ = parse_top3(_completion_text(completion))
        utility = _normalized_utilities(_score_map(raw_scores))
        rewards.append(utility.get(predicted[0], 0.0) if predicted else 0.0)
    return rewards


def calibrated_ranking_reward(
    completions: Sequence[Any],
    reward_by_tariff_id: Sequence[dict[str, float]],
    **_: Any,
) -> list[float]:
    """Discounted utility of a valid top three divided by the ideal top-three utility."""
    rewards: list[float] = []
    for completion, raw_scores in zip(completions, reward_by_tariff_id, strict=True):
        predicted, _ = parse_top3(_completion_text(completion))
        utility = _normalized_utilities(_score_map(raw_scores))
        if len(predicted) != 3 or len(set(predicted)) != 3 or not set(predicted) <= set(utility):
            rewards.append(0.0)
            continue
        ideal = sorted(utility.values(), reverse=True)[:3]
        denominator = sum(weight * value for weight, value in zip(POSITION_WEIGHTS, ideal, strict=True))
        if denominator == 0:
            rewards.append(1.0)
            continue
        numerator = sum(
            weight * utility[tariff_id]
            for weight, tariff_id in zip(POSITION_WEIGHTS, predicted, strict=True)
        )
        rewards.append(numerator / denominator)
    return rewards


def reward_functions() -> list[Callable[..., list[float]]]:
    """Stable ordering corresponding to GRPOConfig.reward_weights."""
    return [
        strict_json_reward,
        valid_tariffs_reward,
        calibrated_top1_reward,
        calibrated_ranking_reward,
    ]
