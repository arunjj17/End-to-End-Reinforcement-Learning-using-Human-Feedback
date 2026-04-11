from __future__ import annotations

import numpy as np


def align_dimensions(prompt: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prompt_vec = np.asarray(prompt, dtype=np.float64)
    response_vec = np.asarray(response, dtype=np.float64)
    target_dim = max(prompt_vec.size, response_vec.size)
    if prompt_vec.size != target_dim:
        prompt_vec = np.resize(prompt_vec, target_dim)
    if response_vec.size != target_dim:
        response_vec = np.resize(response_vec, target_dim)
    return prompt_vec, response_vec


def build_features(prompt: np.ndarray, response: np.ndarray) -> np.ndarray:
    prompt_vec = np.asarray(prompt, dtype=np.float64)
    response_vec = np.asarray(response, dtype=np.float64)
    aligned_prompt, aligned_response = align_dimensions(prompt_vec, response_vec)

    return np.concatenate(
        (
            prompt_vec,
            response_vec,
            aligned_prompt * aligned_response,
            aligned_response**2,
            np.ones(1, dtype=np.float64),
        )
    )
