# LLM Token Pricing Configuration
# Price per 1M tokens (in USD)
# Format: model_name: (input_price, output_price)

import math
import threading
from collections import deque
from typing import List, Optional

LLM_TOKEN_PRICES = {
    # Meta LLaMA Models
    "meta/llama-3.2-3b-instruct": (0.06, 0.06),
    "meta/llama-3.1-8b-instruct": (0.18, 0.18),
    "meta/llama-3.3-70b-instruct": (0.88, 0.88),
    "openai/gpt-oss-120b": (0.15, 0.60),
    # Qwen Models
    "qwen/qwen2.5-7b-instruct": (0.30, 0.30),
    "qwen/qwen2-7b-instruct": (0.30, 0.30),
    'qwen/deepseek-r1-distill-qwen-32b': (0.45, 0.45),
    "qwen/qwen2.5-72b-instruct-turbo": (0.45, 0.45),
    "qwen/qwen2.5-7b-instruct-turbo": (0.30, 0.30),
    "qwen/qwen3-next-80b-a3b-instruct": (0.15, 1.50),
}

def get_token_price(
    model_name: str, token_type: str = "input"
) -> float:
    """
    Get token price for a given model

    Args:
        model_name: Model identifier
            (e.g., "meta/llama-3.2-3b-instruct")
        token_type: "input" or "output"

    Returns:
        Price per 1M tokens (float)
    """
    model_key = model_name.lower()
    if model_key in LLM_TOKEN_PRICES:
        input_price, output_price = LLM_TOKEN_PRICES[model_key]
        return input_price if token_type.lower() == "input" else output_price

    # Fallback: return default price if model not found
    return 0.0


def calculate_cost(
    model_name: str, input_tokens: int, output_tokens: int
) -> float:
    """
    Calculate total cost for a given model and token usage

    Args:
        model_name: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    model_key = model_name.lower()
    if model_key in LLM_TOKEN_PRICES:
        input_price, output_price = LLM_TOKEN_PRICES[model_key]
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

    return 0.0


# ============================================================================
# Cost Normalization
# ============================================================================

class CostNormalizer:
    """
    Cost normalizer that maintains a sliding window of costs and
    normalizes them using percentile-based scaling.

    Process:
    1. Store raw costs in a fixed-size queue (max 2000)
    2. Apply square root transformation to costs
    3. Calculate 95th percentile max and 5th percentile min
    4. Normalize: (sqrt_cost - cost_min) / (cost_max - cost_min)
    5. Clip to [0, 1] range
    """

    def __init__(self, max_size: int = 20000):
        """
        Initialize the cost normalizer

        Args:
            max_size: Maximum number of costs to maintain (default: 2000)
        """
        self.max_size = max_size
        self.cost_queue = deque(maxlen=max_size)
        self._lock = threading.Lock()  # Thread lock, protects shared state

    def add_cost(self, raw_cost: float) -> None:
        """
        Add a raw cost to the queue (thread-safe)

        Args:
            raw_cost: Raw cost value (must be >= 0)
        """
        if raw_cost < 0:
            raw_cost = 0.0
        with self._lock:
            self.cost_queue.append(raw_cost)

    def _calculate_percentiles(self) -> tuple[float, float]:
        """
        Calculate 5th percentile min and 95th percentile max
        from square root transformed costs (thread-safe)

        Returns:
            (cost_min, cost_max) tuple
        """
        # Thread-safely copy queue contents (fast operation, reduce lock hold time)
        with self._lock:
            queue_size = len(self.cost_queue)
            if queue_size == 0:
                return 0.0, 1.0
            # Quickly copy queue, avoid queue being modified during calculation
            costs_copy = list(self.cost_queue)

        # Calculate outside lock (sorting and percentile calculation), reduce lock hold time
        # Apply square root transformation
        sqrt_costs = [math.sqrt(cost) for cost in costs_copy]
        sqrt_costs_sorted = sorted(sqrt_costs)

        n = len(sqrt_costs_sorted)

        # Calculate 5th percentile (min) and 95th percentile (max)
        # Using linear interpolation for percentiles
        idx_5 = 0.05 * (n - 1)
        idx_95 = 0.95 * (n - 1)

        # Linear interpolation
        if n == 1:
            cost_min = cost_max = sqrt_costs_sorted[0]
        else:
            # 5th percentile (min)
            lower_idx_5 = int(math.floor(idx_5))
            upper_idx_5 = int(math.ceil(idx_5))
            if lower_idx_5 == upper_idx_5:
                cost_min = sqrt_costs_sorted[lower_idx_5]
            else:
                weight = idx_5 - lower_idx_5
                cost_min = (
                    sqrt_costs_sorted[lower_idx_5] * (1 - weight) +
                    sqrt_costs_sorted[upper_idx_5] * weight
                )

            # 95th percentile (max)
            lower_idx_95 = int(math.floor(idx_95))
            upper_idx_95 = int(math.ceil(idx_95))
            if lower_idx_95 == upper_idx_95:
                cost_max = sqrt_costs_sorted[lower_idx_95]
            else:
                weight = idx_95 - lower_idx_95
                cost_max = (
                    sqrt_costs_sorted[lower_idx_95] * (1 - weight) +
                    sqrt_costs_sorted[upper_idx_95] * weight
                )

        # Ensure cost_max > cost_min to avoid division by zero
        if cost_max <= cost_min:
            if cost_max == 0:
                cost_max = 1.0
            else:
                cost_min = cost_max * 0.9

        return cost_min, cost_max

    def normalize(self, raw_cost: float) -> float:
        """
        Normalize a raw cost value to [0, 1] range (thread-safe)

        Process:
        1. Apply square root transformation
        2. Calculate percentile-based min/max
        3. Normalize: (sqrt_cost - cost_min) / (cost_max - cost_min)
        4. Clip to [0, 1]

        Args:
            raw_cost: Raw cost value to normalize

        Returns:
            Normalized cost in [0, 1] range
        """
        if raw_cost < 0:
            raw_cost = 0.0

        # Step 1: Apply square root transformation
        sqrt_cost = math.sqrt(raw_cost)

        # Step 2: Calculate percentile-based min/max (dynamic calculation, ensure using latest data)
        cost_min, cost_max = self._calculate_percentiles()

        # Step 3: Normalize
        if cost_max <= cost_min:
            # Edge case: all costs are the same or invalid
            normalized = 0.5
        else:
            normalized = (
                (sqrt_cost - cost_min) /
                (cost_max - cost_min)
            )

        # Step 4: Clip to [0, 1]
        normalized = max(0.0, min(1.0, normalized))

        return normalized

    def get_stats(self) -> dict:
        """
        Get statistics about the current cost distribution (thread-safe)

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            queue_size = len(self.cost_queue)
            if queue_size == 0:
                return {
                    "count": 0,
                    "cost_min": None,
                    "cost_max": None,
                    "sqrt_cost_min": None,
                    "sqrt_cost_max": None,
                }
            costs = list(self.cost_queue)

        cost_min, cost_max = self._calculate_percentiles()

        return {
            "count": len(costs),
            "cost_min": min(costs),
            "cost_max": max(costs),
            "sqrt_cost_min": cost_min,
            "sqrt_cost_max": cost_max,
        }


# Global cost normalizer instance
_cost_normalizer = CostNormalizer(max_size=2000)


def normalize_cost(raw_cost: float, add_to_history: bool = True) -> float:
    """
    Normalize a raw cost value using the global cost normalizer (thread-safe)

    Args:
        raw_cost: Raw cost value to normalize
        add_to_history: Whether to add this cost to the history queue

    Returns:
        Normalized cost in [0, 1] range

    Note:
        This function is thread-safe and can be called concurrently from
        multiple threads. The normalization uses a sliding window of costs
        and applies percentile-based scaling.
    """
    if add_to_history:
        _cost_normalizer.add_cost(raw_cost)
    return _cost_normalizer.normalize(raw_cost)


def get_cost_normalizer_stats() -> dict:
    """
    Get statistics from the global cost normalizer

    Returns:
        Dictionary with cost statistics
    """
    return _cost_normalizer.get_stats()


def reset_cost_normalizer() -> None:
    """
    Reset the global cost normalizer (clear all history)
    """
    global _cost_normalizer
    _cost_normalizer = CostNormalizer(max_size=2000)


def normalize_costs_batch(raw_costs: List[float], add_to_history: bool = True) -> List[float]:
    """
    Normalize a batch of raw cost values using the global cost normalizer (thread-safe)

    Args:
        raw_costs: List of raw cost values to normalize
        add_to_history: Whether to add these costs to the history queue

    Returns:
        List of normalized costs in [0, 1] range

    Note:
        This function is thread-safe and can be called concurrently from
        multiple threads. The normalization uses a sliding window of costs
        and applies percentile-based scaling.
    """
    if not raw_costs:
        return []

    # Add all costs to history if requested
    if add_to_history:
        for raw_cost in raw_costs:
            _cost_normalizer.add_cost(raw_cost)

    # Normalize each cost
    normalized_costs = [_cost_normalizer.normalize(raw_cost) for raw_cost in raw_costs]

    # Convert to cost efficiency (1 - normalized_cost): lower cost = higher efficiency
    cost_efficiencies = [1.0 - nc for nc in normalized_costs]

    return cost_efficiencies


# ============================================================================
# F1 Score Normalization
# ============================================================================

class F1ScoreNormalizer:
    """
    F1 Score normalizer that maintains a sliding window of F1 scores and
    normalizes them using percentile-based scaling.

    Process:
    1. Store raw F1 scores in a fixed-size queue (default 20000)
    2. Calculate percentile-based statistics
    3. Normalize F1 scores to [0, 1] range (higher F1 = higher normalized value)
    4. Provide normalized F1 scores for reward calculation
    """

    def __init__(self, max_size: int = 20000):
        """
        Initialize the F1 score normalizer

        Args:
            max_size: Maximum number of F1 scores to maintain (default: 20000)
        """
        self.max_size = max_size
        self.f1_queue = deque(maxlen=max_size)
        self._lock = threading.Lock()  # Thread lock, protects shared state

    def add_f1_score(self, f1_score: float) -> None:
        """
        Add an F1 score to the queue (thread-safe)

        Args:
            f1_score: F1 score value (typically in [0, 1] range)
        """
        # Ensure F1 score is in valid range
        f1_score = max(0.0, min(1.0, f1_score))

        with self._lock:
            self.f1_queue.append(f1_score)

    def normalize_f1_batch(self, f1_scores: List[float], add_to_history: bool = True) -> List[float]:
        """
        Normalize a batch of F1 scores using sliding window statistics

        Args:
            f1_scores: List of raw F1 scores to normalize
            add_to_history: Whether to add these scores to the history queue

        Returns:
            List of normalized F1 scores in [0, 1] range
        """
        if not f1_scores:
            return []

        # Add all F1 scores to history if requested
        if add_to_history:
            for f1_score in f1_scores:
                self.add_f1_score(f1_score)

        # Normalize each F1 score
        normalized_f1_scores = [self.normalize(f1_score) for f1_score in f1_scores]

        return normalized_f1_scores

    def normalize(self, f1_score: float) -> float:
        """
        Normalize a single F1 score to [0, 1] range (thread-safe)

        Process:
        1. Ensure F1 score is in [0, 1] range
        2. Calculate percentile-based statistics from sliding window
        3. Normalize: (f1_score - f1_min) / (f1_max - f1_min)
        4. Clip to [0, 1] range

        Args:
            f1_score: Raw F1 score to normalize

        Returns:
            Normalized F1 score in [0, 1] range
        """
        # Ensure F1 score is in valid range
        f1_score = max(0.0, min(1.0, f1_score))

        with self._lock:
            if len(self.f1_queue) == 0:
                # No history available, return the score as-is
                return f1_score

            # Calculate percentile-based statistics
            sorted_f1_scores = sorted(self.f1_queue)
            n = len(sorted_f1_scores)

            if n < 10:
                # Too few samples, use simple min/max
                f1_min = min(sorted_f1_scores)
                f1_max = max(sorted_f1_scores)
            else:
                # Use percentile-based approach
                idx_5 = max(0, int(0.05 * (n - 1)))
                idx_95 = min(n - 1, int(0.95 * (n - 1)))
                f1_min = sorted_f1_scores[idx_5]
                f1_max = sorted_f1_scores[idx_95]

            # Normalize
            if f1_max <= f1_min:
                # Edge case: all F1 scores are the same
                normalized = 0.5
            else:
                normalized = (f1_score - f1_min) / (f1_max - f1_min)

            # Clip to [0, 1] range
            normalized = max(0.0, min(1.0, normalized))

            return normalized

    def get_f1_stats(self) -> dict:
        """
        Get statistics about the F1 score sliding window

        Returns:
            Dictionary with F1 score statistics
        """
        with self._lock:
            if len(self.f1_queue) == 0:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0
                }

            sorted_scores = sorted(self.f1_queue)
            n = len(sorted_scores)

            return {
                'count': n,
                'mean': sum(self.f1_queue) / n,
                'std': math.sqrt(sum((x - sum(self.f1_queue) / n) ** 2 for x in self.f1_queue) / n) if n > 1 else 0.0,
                'min': sorted_scores[0],
                'max': sorted_scores[-1],
                'median': sorted_scores[n // 2]
            }


# Global F1 score normalizer instance
_f1_normalizer = F1ScoreNormalizer(max_size=20000)

# ============================================================================
# Reward-Cost Scale Alignment
# ============================================================================

import numpy as np
from typing import Tuple, Optional

# Global history for reward-cost scale alignment
_global_historical_f1_rewards = []
_global_historical_costs = []
_max_history_size = 20000  # Match F1 normalizer size


def add_to_scale_alignment_history(f1_reward: float, cost: float) -> None:
    """
    Add F1 reward and cost to the global history for scale alignment calculations.

    Args:
        f1_reward: F1 reward value (normalized or raw)
        cost: Cost value (normalized or raw)
    """
    global _global_historical_f1_rewards, _global_historical_costs

    _global_historical_f1_rewards.append(f1_reward)
    _global_historical_costs.append(cost)

    # Maintain fixed-size history
    if len(_global_historical_f1_rewards) > _max_history_size:
        _global_historical_f1_rewards.pop(0)
        _global_historical_costs.pop(0)


def get_scale_alignment_stats() -> dict:
    """
    Get statistics about the current scale alignment history.

    Returns:
        Dictionary with scale alignment statistics
    """
    global _global_historical_f1_rewards, _global_historical_costs

    n = len(_global_historical_f1_rewards)

    if n == 0:
        return {
            'count': 0,
            'f1_mean': 0.0,
            'f1_std': 0.0,
            'cost_mean': 0.0,
            'cost_std': 0.0,
            'scale_ratio': 1.0
        }

    f1_rewards = np.array(_global_historical_f1_rewards)
    costs = np.array(_global_historical_costs)

    f1_std = np.std(f1_rewards) if n > 1 else 0.1  # Avoid division by zero
    cost_std = np.std(costs) if n > 1 else 0.01   # Avoid division by zero

    # Avoid zero standard deviation
    f1_std = max(f1_std, 0.001)
    cost_std = max(cost_std, 0.001)

    scale_ratio = cost_std / f1_std

    return {
        'count': n,
        'f1_mean': float(np.mean(f1_rewards)),
        'f1_std': float(f1_std),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(cost_std),
        'scale_ratio': float(scale_ratio)
    }


def align_reward_cost_scales_batch(
    f1_rewards: List[float],
    costs: List[float],
    target_reward_scale: float = 1.0,
    historical_f1_rewards: Optional[list] = None,
    historical_costs: Optional[list] = None,
    add_to_global_history: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Batch version of align_reward_cost_scales for processing multiple reward-cost pairs.

    Process:
    1. Add current batch data to global history (if add_to_global_history=True)
    2. Calculate scaling statistics from updated history (including current batch)
    3. Apply the same scale factor to all items in the batch
    4. Return scaled rewards and original costs

    Args:
        f1_rewards: List of F1 reward values to be scaled
        costs: List of corresponding cost values (remain unchanged)
        target_reward_scale: Additional scaling factor (default: 1.0)
        historical_f1_rewards: Optional custom history for F1 rewards (if provided, 
                              will use this instead of global history)
        historical_costs: Optional custom history for costs (if provided, 
                         will use this instead of global history)
        add_to_global_history: Whether to add current batch to global history before scaling.
                              If False, only uses existing history. Default: True.

    Returns:
        Tuple of (scaled_f1_rewards, original_costs)
    """
    if len(f1_rewards) != len(costs):
        raise ValueError("f1_rewards and costs must have the same length")

    if not f1_rewards:
        return [], []

    # Step 1: Add current batch data to global history (if using global history)
    if add_to_global_history and historical_f1_rewards is None and historical_costs is None:
        global _global_historical_f1_rewards, _global_historical_costs, _max_history_size
        
        # Add all items in the batch to global history
        for f1_reward, cost in zip(f1_rewards, costs):
            _global_historical_f1_rewards.append(f1_reward)
            _global_historical_costs.append(cost)
            
            # Maintain fixed-size history
            if len(_global_historical_f1_rewards) > _max_history_size:
                _global_historical_f1_rewards.pop(0)
                _global_historical_costs.pop(0)

    # Step 2: Get the appropriate historical data (now includes current batch if added)
    if historical_f1_rewards is not None and historical_costs is not None:
        # Use provided custom history
        f1_history = historical_f1_rewards + f1_rewards  # Include current batch
        cost_history = historical_costs + costs  # Include current batch
    else:
        # Use global history (which now includes current batch if add_to_global_history=True)
        f1_history = _global_historical_f1_rewards
        cost_history = _global_historical_costs

    # Step 3: Check if we have enough history for meaningful scaling
    if len(f1_history) < 2 or len(cost_history) < 2:
        # Not enough history, return original values with target scaling only
        scale_factor = target_reward_scale
    else:
        # Step 4: Calculate standard deviations from historical data (including current batch)
        f1_std = np.std(f1_history) if len(f1_history) > 1 else 0.1
        cost_std = np.std(cost_history) if len(cost_history) > 1 else 0.01

        # Step 5: Avoid division by zero or near-zero values
        f1_std = max(f1_std, 0.00001)
        cost_std = max(cost_std, 0.00001)

        # Step 6: Calculate scale factor to align standard deviations
        # This aligns F1 reward distribution to match cost distribution
        scale_factor = (cost_std / f1_std) * target_reward_scale

    # Step 7: Apply the same scale factor to all items in batch
    scaled_f1_rewards = [f1 * scale_factor for f1 in f1_rewards]

    # Step 8: Return scaled rewards and original costs (costs remain unchanged)
    return scaled_f1_rewards, costs


def align_reward_cost_scales(
    f1_reward: float,
    cost: float,
    target_reward_scale: float = 1.0,
    historical_f1_rewards: Optional[list] = None,
    historical_costs: Optional[list] = None
) -> Tuple[float, float]:
    """
    Align the scale of F1 reward to match the cost scale using historical statistics.

    Mathematical Principle:
    The goal is to make F1 reward changes comparable to cost changes in magnitude.
    We use the ratio of historical standard deviations to determine the appropriate scaling.

    Process:
    1. Calculate standard deviations from historical F1 rewards and costs
    2. Compute scale factor: scale_factor = cost_std / f1_std * target_reward_scale
    3. Apply scaling: scaled_f1 = f1_reward * scale_factor
    4. Return scaled F1 reward and original cost

    Args:
        f1_reward: Current F1 reward value to be scaled
        cost: Current cost value (remains unchanged)
        target_reward_scale: Additional scaling factor (default: 1.0)
        historical_f1_rewards: Optional custom history for F1 rewards
        historical_costs: Optional custom history for costs

    Returns:
        Tuple of (scaled_f1_reward, original_cost)
    """
    # Use provided history or global history
    if historical_f1_rewards is None or historical_costs is None:
        f1_history = _global_historical_f1_rewards
        cost_history = _global_historical_costs
    else:
        f1_history = historical_f1_rewards
        cost_history = historical_costs

    # Check if we have enough history for meaningful scaling
    if len(f1_history) < 2 or len(cost_history) < 2:
        # Not enough history, return original values with target scaling
        return f1_reward * target_reward_scale, cost

    # Calculate standard deviations
    f1_std = np.std(f1_history) if len(f1_history) > 1 else 0.1
    cost_std = np.std(cost_history) if len(cost_history) > 1 else 0.01

    # Avoid division by zero or near-zero values
    f1_std = max(f1_std, 0.001)
    cost_std = max(cost_std, 0.001)

    # Calculate scale factor to align standard deviations
    scale_factor = (cost_std / f1_std) * target_reward_scale

    # Apply scaling to F1 reward
    scaled_f1 = f1_reward * scale_factor

    # Return scaled F1 reward and original cost
    return scaled_f1, cost


def reset_scale_alignment_history() -> None:
    """
    Reset the global scale alignment history.
    """
    global _global_historical_f1_rewards, _global_historical_costs
    _global_historical_f1_rewards = []
    _global_historical_costs = []


def normalize_f1_scores_batch(f1_scores: List[float], add_to_history: bool = True) -> List[float]:
    """
    Normalize a batch of F1 scores using sliding window normalization

    Args:
        f1_scores: List of raw F1 scores to normalize
        add_to_history: Whether to add these scores to the history queue

    Returns:
        List of normalized F1 scores in [0, 1] range
    """
    return _f1_normalizer.normalize_f1_batch(f1_scores, add_to_history)


def get_f1_normalizer_stats() -> dict:
    """
    Get statistics about the global F1 score normalizer

    Returns:
        Dictionary with F1 score normalizer statistics
    """
    return _f1_normalizer.get_f1_stats()

