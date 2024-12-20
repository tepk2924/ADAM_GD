# pso_optimization.py

import numpy as np
from typing import Callable, Tuple, List

def optimize(
    cost_function: Callable[[List[float]], float],
    n_dimensions: int,
    boundaries: List[Tuple[float, float]],
    n_iterations: int,
    learning_rate: float,
    beta_1: float,
    beta_2: float,
    delta: float,
    verbose: bool = False,
    callback: Callable[[List[float], int], None] = None
) -> Tuple[List[float], float, List[List[float]], List[float]]:

    #도로 중앙 (0.5)을 달리는 것으로 초기화
    position = np.full((n_dimensions), 0.5, dtype=float)
    m = np.zeros((n_dimensions), dtype=float)
    v = np.zeros((n_dimensions), dtype=float)
    epsilon = 1e-8

    score_curr = cost_function(position.tolist())
    
    if verbose:
        print(f"Initial score: {score_curr}")
    
    d_mat = delta*np.eye(n_dimensions, dtype=float)

    evaluation_history = []

    #본 계산 작업
    try:
        for iteration in range(n_iterations):

            #경사값 구하기
            score_plus_epsilon = np.array([cost_function((position + d_mat[idx]).tolist()) for idx in range(n_dimensions)])
            gradient = (score_plus_epsilon - score_curr)/delta
            m = beta_1*m + (1 - beta_1)*gradient
            v = beta_2*v + (1 - beta_2)*(gradient**2)
            m_hat = m/(1 - beta_1**(iteration + 1))
            v_hat = v/(1 - beta_2**(iteration + 1))

            #경사 하강

            position -= m_hat*learning_rate/(np.sqrt(v_hat) + epsilon)
            position = np.clip(position, [b[0] for b in boundaries], [b[1] for b in boundaries])

            score_curr = cost_function(position.tolist())

            if callback:
                callback(position, iteration)

            if verbose:
                print(f"Iteration {iteration + 1}/{n_iterations}, Current Score: {score_curr}")

            evaluation_history.append(score_curr)
    except KeyboardInterrupt:
        pass

    return list(position), score_curr, None, evaluation_history
