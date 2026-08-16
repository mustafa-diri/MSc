import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

NUMBER_OF_SOURCES = 100
NUMBER_OF_DESTINATIONS = 100

PROBLEM_SEED = 2026
BASE_ALGORITHM_SEED = 5000

POPULATION_SIZE = 20
MAX_ITERATIONS = 60
INDEPENDENT_RUNS = 30

DINKELBACH_RELATIVE_TOLERANCE = 1e-9
DINKELBACH_MAX_ITERATIONS = 100
FEASIBILITY_TOLERANCE = 1e-6

OUTPUT_DIRECTORY = Path(r"D:\mustafa paper")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

@dataclass
class Result:
    method: str
    X: np.ndarray
    objective: float
    numerator: float
    denominator: float
    elapsed_time: float
    feasibility_error: float
    convergence: list[float]
    function_evaluations: int


def generate_balanced_problem(
    number_of_sources: int,
    number_of_destinations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    supply = rng.integers(
        500,
        2001,
        size=number_of_sources,
    ).astype(float)

    total_supply = int(supply.sum())

    demand_weights = rng.random(number_of_destinations)
    demand_weights /= demand_weights.sum()

    raw_demand = demand_weights * total_supply
    demand = np.floor(raw_demand).astype(float)

    remaining_units = int(total_supply - demand.sum())

    fractional_parts = raw_demand - np.floor(raw_demand)
    correction_order = np.argsort(-fractional_parts)

    for j in correction_order[:remaining_units]:
        demand[j] += 1.0

    C = rng.uniform(
        10.0,
        150.0,
        size=(number_of_sources, number_of_destinations),
    )

    D = rng.uniform(
        1.0,
        30.0,
        size=(number_of_sources, number_of_destinations),
    )

    return C, D, supply, demand


C, D, supply, demand = generate_balanced_problem(
    NUMBER_OF_SOURCES,
    NUMBER_OF_DESTINATIONS,
    PROBLEM_SEED,
)

m, n = C.shape
dimension = m * n

c0 = 0.0
d0 = 1.0

def numerator(X: np.ndarray) -> float:
    return float(c0 + np.sum(C * X))

def denominator(X: np.ndarray) -> float:
    return float(d0 + np.sum(D * X))

def fractional_objective(X: np.ndarray) -> float:
    g_value = denominator(X)

    if not np.isfinite(g_value) or g_value <= 0.0:
        return np.inf

    f_value = numerator(X)

    if not np.isfinite(f_value):
        return np.inf

    return f_value / g_value


def calculate_feasibility_error(X: np.ndarray) -> float:
    if X.shape != (m, n):
        return np.inf

    if not np.all(np.isfinite(X)):
        return np.inf

    row_error = float(
        np.max(
            np.abs(X.sum(axis=1) - supply)
        )
    )

    column_error = float(
        np.max(
            np.abs(X.sum(axis=0) - demand)
        )
    )

    negative_error = max(
        0.0,
        float(-np.min(X)),
    )

    return max(
        row_error,
        column_error,
        negative_error,
    )


def is_feasible(
    X: np.ndarray,
    tolerance: float = FEASIBILITY_TOLERANCE,
) -> bool:
    return calculate_feasibility_error(X) <= tolerance


def decode_random_keys(keys: np.ndarray) -> np.ndarray:
    keys_array = np.asarray(keys, dtype=float)

    if keys_array.size != dimension:
        raise ValueError(
            f"Expected {dimension} keys, "
            f"but received {keys_array.size}."
        )

    if not np.all(np.isfinite(keys_array)):
        raise ValueError(
            "The random-key vector contains non-finite values."
        )

    priority_order = np.argsort(
        -keys_array,
        kind="stable",
    )

    remaining_supply = supply.copy()
    remaining_demand = demand.copy()

    X = np.zeros((m, n), dtype=float)
    remaining_total = float(supply.sum())

    for flat_index in priority_order:
        if remaining_total <= FEASIBILITY_TOLERANCE:
            break

        i = int(flat_index // n)
        j = int(flat_index % n)

        if remaining_supply[i] <= FEASIBILITY_TOLERANCE:
            continue

        if remaining_demand[j] <= FEASIBILITY_TOLERANCE:
            continue

        quantity = min(
            remaining_supply[i],
            remaining_demand[j],
        )

        if quantity <= 0.0:
            continue

        X[i, j] += quantity

        remaining_supply[i] -= quantity
        remaining_demand[j] -= quantity
        remaining_total -= quantity

        if abs(remaining_supply[i]) <= FEASIBILITY_TOLERANCE:
            remaining_supply[i] = 0.0

        if abs(remaining_demand[j]) <= FEASIBILITY_TOLERANCE:
            remaining_demand[j] = 0.0

    active_sources = np.where(
        remaining_supply > FEASIBILITY_TOLERANCE
    )[0]

    active_destinations = np.where(
        remaining_demand > FEASIBILITY_TOLERANCE
    )[0]

    source_pointer = 0
    destination_pointer = 0

    while (
        source_pointer < len(active_sources)
        and destination_pointer < len(active_destinations)
    ):
        i = int(active_sources[source_pointer])
        j = int(active_destinations[destination_pointer])

        quantity = min(
            remaining_supply[i],
            remaining_demand[j],
        )

        X[i, j] += quantity

        remaining_supply[i] -= quantity
        remaining_demand[j] -= quantity

        if remaining_supply[i] <= FEASIBILITY_TOLERANCE:
            remaining_supply[i] = 0.0
            source_pointer += 1

        if remaining_demand[j] <= FEASIBILITY_TOLERANCE:
            remaining_demand[j] = 0.0
            destination_pointer += 1

    X[np.abs(X) <= 1e-12] = 0.0

    error = calculate_feasibility_error(X)

    if error > FEASIBILITY_TOLERANCE:
        raise RuntimeError(
            "The decoder generated an infeasible solution. "
            f"Error = {error:.6e}"
        )

    return X


def evaluate_keys(
    keys: np.ndarray,
) -> tuple[float, np.ndarray]:

    X = decode_random_keys(keys)

    return fractional_objective(X), X


def create_heuristic_keys() -> np.ndarray:
    route_ratio = C / D
    score = -route_ratio.ravel()

    score_min = float(score.min())
    score_max = float(score.max())

    if score_max - score_min <= 1e-15:
        return np.full(dimension, 0.5)

    return (
        score - score_min
    ) / (
        score_max - score_min
    )


HEURISTIC_KEYS = create_heuristic_keys()


def build_sparse_transportation_constraints(
) -> tuple[csr_matrix, np.ndarray]:

    rows = []
    columns = []
    values = []
    right_hand_side = []

    constraint_index = 0

    for i in range(m):
        for j in range(n):
            rows.append(constraint_index)
            columns.append(i * n + j)
            values.append(1.0)

        right_hand_side.append(float(supply[i]))
        constraint_index += 1

    for j in range(n - 1):
        for i in range(m):
            rows.append(constraint_index)
            columns.append(i * n + j)
            values.append(1.0)

        right_hand_side.append(float(demand[j]))
        constraint_index += 1

    A_eq = coo_matrix(
        (values, (rows, columns)),
        shape=(m + n - 1, dimension),
    ).tocsr()

    b_eq = np.asarray(
        right_hand_side,
        dtype=float,
    )

    return A_eq, b_eq


A_EQ_X, B_EQ_X = build_sparse_transportation_constraints()


def solve_charnes_cooper() -> Result:
    start_time = time.perf_counter()

    number_of_variables = dimension + 1

    rows = []
    columns = []
    values = []
    right_hand_side = []

    constraint_index = 0

    for i in range(m):
        for j in range(n):
            rows.append(constraint_index)
            columns.append(i * n + j)
            values.append(1.0)

        rows.append(constraint_index)
        columns.append(dimension)
        values.append(-float(supply[i]))

        right_hand_side.append(0.0)
        constraint_index += 1

    for j in range(n - 1):
        for i in range(m):
            rows.append(constraint_index)
            columns.append(i * n + j)
            values.append(1.0)

        rows.append(constraint_index)
        columns.append(dimension)
        values.append(-float(demand[j]))

        right_hand_side.append(0.0)
        constraint_index += 1

    for variable_index, coefficient in enumerate(D.ravel()):
        rows.append(constraint_index)
        columns.append(variable_index)
        values.append(float(coefficient))

    rows.append(constraint_index)
    columns.append(dimension)
    values.append(float(d0))

    right_hand_side.append(1.0)

    A_eq = coo_matrix(
        (values, (rows, columns)),
        shape=(m + n, number_of_variables),
    ).tocsr()

    b_eq = np.asarray(
        right_hand_side,
        dtype=float,
    )

    objective_coefficients = np.concatenate(
        [
            C.ravel(),
            np.array([c0]),
        ]
    )

    optimization_result = linprog(
        c=objective_coefficients,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
        options={"presolve": True},
    )

    if not optimization_result.success:
        raise RuntimeError(
            "Charnes-Cooper failed: "
            + optimization_result.message
        )

    y = optimization_result.x[:-1].reshape(m, n)
    t = float(optimization_result.x[-1])

    if not np.isfinite(t) or t <= 0.0:
        raise RuntimeError(
            f"Charnes-Cooper returned invalid t = {t}"
        )

    X = y / t
    X[np.abs(X) <= 1e-10] = 0.0

    elapsed_time = time.perf_counter() - start_time
    error = calculate_feasibility_error(X)

    if error > 1e-4:
        raise RuntimeError(
            "Charnes-Cooper returned an infeasible solution. "
            f"Error = {error:.6e}"
        )

    return Result(
        method="Charnes-Cooper",
        X=X,
        objective=fractional_objective(X),
        numerator=numerator(X),
        denominator=denominator(X),
        elapsed_time=elapsed_time,
        feasibility_error=error,
        convergence=[fractional_objective(X)],
        function_evaluations=1,
    )


def solve_dinkelbach() -> Result:
    start_time = time.perf_counter()

    lambda_value = 0.0
    convergence = []
    X = np.zeros((m, n), dtype=float)

    linear_program_count = 0

    for iteration in range(DINKELBACH_MAX_ITERATIONS):
        modified_cost = (
            C - lambda_value * D
        ).ravel()

        optimization_result = linprog(
            c=modified_cost,
            A_eq=A_EQ_X,
            b_eq=B_EQ_X,
            bounds=(0.0, None),
            method="highs",
            options={"presolve": True},
        )

        linear_program_count += 1

        if not optimization_result.success:
            raise RuntimeError(
                "Dinkelbach subproblem failed: "
                + optimization_result.message
            )

        X = optimization_result.x.reshape(m, n)
        X[np.abs(X) <= 1e-10] = 0.0

        f_value = numerator(X)
        g_value = denominator(X)

        if g_value <= 0.0:
            raise RuntimeError(
                "Dinkelbach encountered a non-positive denominator."
            )

        residual = (
            f_value
            - lambda_value * g_value
        )

        new_lambda = f_value / g_value
        convergence.append(new_lambda)

        relative_residual = (
            abs(residual)
            / max(
                1.0,
                abs(f_value),
                abs(lambda_value * g_value),
            )
        )

        print(
            f"Dinkelbach iteration {iteration + 1:3d}: "
            f"lambda = {new_lambda:.12f}, "
            f"relative residual = {relative_residual:.6e}"
        )

        lambda_value = new_lambda

        if relative_residual <= DINKELBACH_RELATIVE_TOLERANCE:
            break

    else:
        raise RuntimeError(
            "Dinkelbach did not converge within "
            "the maximum number of iterations."
        )

    elapsed_time = time.perf_counter() - start_time
    error = calculate_feasibility_error(X)

    if error > 1e-4:
        raise RuntimeError(
            "Dinkelbach returned an infeasible solution. "
            f"Error = {error:.6e}"
        )

    return Result(
        method="Dinkelbach",
        X=X,
        objective=fractional_objective(X),
        numerator=numerator(X),
        denominator=denominator(X),
        elapsed_time=elapsed_time,
        feasibility_error=error,
        convergence=convergence,
        function_evaluations=linear_program_count,
    )


def solve_genetic_algorithm(seed: int) -> Result:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    population = rng.random(
        size=(POPULATION_SIZE, dimension)
    )

    population[0] = HEURISTIC_KEYS.copy()

    crossover_probability = 0.90
    mutation_probability = max(
        1.0 / dimension,
        0.002,
    )

    mutation_sigma = 0.15
    elite_size = 2
    function_evaluations = 0

    def evaluate_population(
        current_population: np.ndarray,
    ) -> np.ndarray:

        nonlocal function_evaluations

        result_values = np.empty(
            current_population.shape[0],
            dtype=float,
        )

        for index, individual in enumerate(current_population):
            result_values[index] = evaluate_keys(individual)[0]
            function_evaluations += 1

        return result_values

    values = evaluate_population(population)

    best_index = int(np.argmin(values))
    best_position = population[best_index].copy()
    best_value = float(values[best_index])

    convergence = [best_value]

    def tournament_selection() -> np.ndarray:
        selected_indices = rng.choice(
            POPULATION_SIZE,
            size=min(3, POPULATION_SIZE),
            replace=False,
        )

        winner = selected_indices[
            np.argmin(values[selected_indices])
        ]

        return population[winner].copy()

    for iteration in range(MAX_ITERATIONS):
        order = np.argsort(values)

        new_population = [
            population[index].copy()
            for index in order[:elite_size]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection()
            parent2 = tournament_selection()

            child1 = parent1.copy()
            child2 = parent2.copy()

            if rng.random() < crossover_probability:
                alpha = rng.random(dimension)

                child1 = (
                    alpha * parent1
                    + (1.0 - alpha) * parent2
                )

                child2 = (
                    (1.0 - alpha) * parent1
                    + alpha * parent2
                )

            mutation_mask1 = (
                rng.random(dimension)
                < mutation_probability
            )

            mutation_mask2 = (
                rng.random(dimension)
                < mutation_probability
            )

            if np.any(mutation_mask1):
                child1[mutation_mask1] += rng.normal(
                    0.0,
                    mutation_sigma,
                    size=int(mutation_mask1.sum()),
                )

            if np.any(mutation_mask2):
                child2[mutation_mask2] += rng.normal(
                    0.0,
                    mutation_sigma,
                    size=int(mutation_mask2.sum()),
                )

            child1 = np.clip(child1, 0.0, 1.0)
            child2 = np.clip(child2, 0.0, 1.0)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = np.asarray(
            new_population,
            dtype=float,
        )

        values = evaluate_population(population)

        iteration_best_index = int(np.argmin(values))
        iteration_best_value = float(
            values[iteration_best_index]
        )

        if iteration_best_value < best_value:
            best_value = iteration_best_value
            best_position = population[
                iteration_best_index
            ].copy()

        convergence.append(best_value)

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"GA iteration {iteration + 1:3d}: "
                f"best Z = {best_value:.12f}"
            )

    X = decode_random_keys(best_position)

    return Result(
        method="Genetic Algorithm",
        X=X,
        objective=fractional_objective(X),
        numerator=numerator(X),
        denominator=denominator(X),
        elapsed_time=time.perf_counter() - start_time,
        feasibility_error=calculate_feasibility_error(X),
        convergence=convergence,
        function_evaluations=function_evaluations,
    )


def solve_particle_swarm(seed: int) -> Result:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    positions = rng.random(
        size=(POPULATION_SIZE, dimension)
    )

    positions[0] = HEURISTIC_KEYS.copy()

    velocities = rng.uniform(
        -0.05,
        0.05,
        size=(POPULATION_SIZE, dimension),
    )

    function_evaluations = 0

    def evaluate_positions(
        current_positions: np.ndarray,
    ) -> np.ndarray:

        nonlocal function_evaluations

        result_values = np.empty(
            current_positions.shape[0],
            dtype=float,
        )

        for index, position in enumerate(current_positions):
            result_values[index] = evaluate_keys(position)[0]
            function_evaluations += 1

        return result_values

    values = evaluate_positions(positions)

    personal_best_positions = positions.copy()
    personal_best_values = values.copy()

    global_best_index = int(
        np.argmin(personal_best_values)
    )

    global_best_position = personal_best_positions[
        global_best_index
    ].copy()

    global_best_value = float(
        personal_best_values[global_best_index]
    )

    convergence = [global_best_value]

    c1 = 1.7
    c2 = 1.7
    velocity_limit = 0.20

    for iteration in range(MAX_ITERATIONS):
        inertia = (
            0.90
            - 0.50
            * iteration
            / max(1, MAX_ITERATIONS - 1)
        )

        r1 = rng.random(
            size=(POPULATION_SIZE, dimension)
        )

        r2 = rng.random(
            size=(POPULATION_SIZE, dimension)
        )

        velocities = (
            inertia * velocities
            + c1
            * r1
            * (
                personal_best_positions
                - positions
            )
            + c2
            * r2
            * (
                global_best_position
                - positions
            )
        )

        velocities = np.clip(
            velocities,
            -velocity_limit,
            velocity_limit,
        )

        positions = np.clip(
            positions + velocities,
            0.0,
            1.0,
        )

        values = evaluate_positions(positions)

        improved = values < personal_best_values

        personal_best_positions[improved] = (
            positions[improved]
        )

        personal_best_values[improved] = (
            values[improved]
        )

        candidate_index = int(
            np.argmin(personal_best_values)
        )

        candidate_value = float(
            personal_best_values[candidate_index]
        )

        if candidate_value < global_best_value:
            global_best_value = candidate_value
            global_best_position = (
                personal_best_positions[
                    candidate_index
                ].copy()
            )

        convergence.append(global_best_value)

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"PSO iteration {iteration + 1:3d}: "
                f"best Z = {global_best_value:.12f}"
            )

    X = decode_random_keys(global_best_position)

    return Result(
        method="Particle Swarm Optimization",
        X=X,
        objective=fractional_objective(X),
        numerator=numerator(X),
        denominator=denominator(X),
        elapsed_time=time.perf_counter() - start_time,
        feasibility_error=calculate_feasibility_error(X),
        convergence=convergence,
        function_evaluations=function_evaluations,
    )


def solve_differential_evolution(seed: int) -> Result:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    if POPULATION_SIZE < 4:
        raise ValueError(
            "Differential Evolution requires at least four individuals."
        )

    population = rng.random(
        size=(POPULATION_SIZE, dimension)
    )

    population[0] = HEURISTIC_KEYS.copy()

    function_evaluations = 0

    def evaluate_one(
        individual: np.ndarray,
    ) -> float:

        nonlocal function_evaluations
        function_evaluations += 1
        return evaluate_keys(individual)[0]

    values = np.array(
        [
            evaluate_one(individual)
            for individual in population
        ],
        dtype=float,
    )

    best_index = int(np.argmin(values))
    best_position = population[best_index].copy()
    best_value = float(values[best_index])

    convergence = [best_value]

    mutation_factor = 0.60
    crossover_rate = 0.90
    all_indices = np.arange(POPULATION_SIZE)

    for iteration in range(MAX_ITERATIONS):
        next_population = population.copy()
        next_values = values.copy()

        for target_index in range(POPULATION_SIZE):
            available_indices = np.delete(
                all_indices,
                target_index,
            )

            r1_index, r2_index, r3_index = rng.choice(
                available_indices,
                size=3,
                replace=False,
            )

            mutant = (
                population[r1_index]
                + mutation_factor
                * (
                    population[r2_index]
                    - population[r3_index]
                )
            )

            mutant = np.clip(mutant, 0.0, 1.0)

            crossover_mask = (
                rng.random(dimension)
                <= crossover_rate
            )

            crossover_mask[
                int(rng.integers(dimension))
            ] = True

            trial = np.where(
                crossover_mask,
                mutant,
                population[target_index],
            )

            trial_value = evaluate_one(trial)

            if trial_value < values[target_index]:
                next_population[target_index] = trial
                next_values[target_index] = trial_value

        population = next_population
        values = next_values

        iteration_best_index = int(np.argmin(values))
        iteration_best_value = float(
            values[iteration_best_index]
        )

        if iteration_best_value < best_value:
            best_value = iteration_best_value
            best_position = population[
                iteration_best_index
            ].copy()

        convergence.append(best_value)

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"DE iteration {iteration + 1:3d}: "
                f"best Z = {best_value:.12f}"
            )

    X = decode_random_keys(best_position)

    return Result(
        method="Differential Evolution",
        X=X,
        objective=fractional_objective(X),
        numerator=numerator(X),
        denominator=denominator(X),
        elapsed_time=time.perf_counter() - start_time,
        feasibility_error=calculate_feasibility_error(X),
        convergence=convergence,
        function_evaluations=function_evaluations,
    )


def solve_grey_wolf(seed: int) -> Result:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    if POPULATION_SIZE < 3:
        raise ValueError(
            "Grey Wolf Optimizer requires at least three wolves."
        )

    wolves = rng.random(
        size=(POPULATION_SIZE, dimension)
    )

    wolves[0] = HEURISTIC_KEYS.copy()
    function_evaluations = 0

    def evaluate_wolves(
        current_wolves: np.ndarray,
    ) -> np.ndarray:

        nonlocal function_evaluations

        result_values = np.empty(
            current_wolves.shape[0],
            dtype=float,
        )

        for index, wolf in enumerate(current_wolves):
            result_values[index] = evaluate_keys(wolf)[0]
            function_evaluations += 1

        return result_values

    values = evaluate_wolves(wolves)
    order = np.argsort(values)

    alpha_position = wolves[order[0]].copy()
    beta_position = wolves[order[1]].copy()
    delta_position = wolves[order[2]].copy()

    best_position = alpha_position.copy()
    best_value = float(values[order[0]])

    convergence = [best_value]

    for iteration in range(MAX_ITERATIONS):
        a = (
            2.0
            - 2.0
            * iteration
            / max(1, MAX_ITERATIONS - 1)
        )

        new_wolves = np.empty_like(wolves)

        for wolf_index in range(POPULATION_SIZE):
            current_position = wolves[wolf_index]
            proposed_positions = []

            for leader_position in (
                alpha_position,
                beta_position,
                delta_position,
            ):
                r1 = rng.random(dimension)
                r2 = rng.random(dimension)

                A_vector = 2.0 * a * r1 - a
                C_vector = 2.0 * r2

                distance = np.abs(
                    C_vector * leader_position
                    - current_position
                )

                proposed_positions.append(
                    leader_position
                    - A_vector * distance
                )

            new_wolves[wolf_index] = np.clip(
                np.mean(proposed_positions, axis=0),
                0.0,
                1.0,
            )

        new_wolves[0] = best_position.copy()

        wolves = new_wolves
        values = evaluate_wolves(wolves)
        order = np.argsort(values)

        alpha_position = wolves[order[0]].copy()
        beta_position = wolves[order[1]].copy()
        delta_position = wolves[order[2]].copy()

        alpha_value = float(values[order[0]])

        if alpha_value < best_value:
            best_value = alpha_value
            best_position = alpha_position.copy()

        convergence.append(best_value)

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"GWO iteration {iteration + 1:3d}: "
                f"best Z = {best_value:.12f}"
            )

    X = decode_random_keys(best_position)

    return Result(
        method="Grey Wolf Optimizer",
        X=X,
        objective=fractional_objective(X),
        numerator=numerator(X),
        denominator=denominator(X),
        elapsed_time=time.perf_counter() - start_time,
        feasibility_error=calculate_feasibility_error(X),
        convergence=convergence,
        function_evaluations=function_evaluations,
    )


def run_repeated(
    solver: Callable[[int], Result],
    method_name: str,
    seed_offset: int,
) -> tuple[Result, list[Result]]:

    results = []

    for run_index in range(INDEPENDENT_RUNS):
        seed = (
            BASE_ALGORITHM_SEED
            + seed_offset
            + run_index
        )

        print("\n" + "-" * 75)
        print(
            f"{method_name}: "
            f"run {run_index + 1}/{INDEPENDENT_RUNS}"
        )
        print("-" * 75)

        result = solver(seed)
        results.append(result)

        print(
            f"Final Z              = {result.objective:.12f}"
        )

        print(
            f"Execution time       = "
            f"{result.elapsed_time:.3f} seconds"
        )

        print(
            f"Feasibility error    = "
            f"{result.feasibility_error:.6e}"
        )

        print(
            f"Function evaluations = "
            f"{result.function_evaluations:,}"
        )

    best_result = min(
        results,
        key=lambda item: item.objective,
    )

    return best_result, results


def calculate_gap(
    value: float,
    comparison_value: float,
) -> float:

    return (
        100.0
        * abs(value - comparison_value)
        / max(abs(comparison_value), 1e-15)
    )


def summarize_stochastic_results(
    method_name: str,
    results: list[Result],
    charnes_cooper_value: float,
    dinkelbach_value: float,
) -> dict:

    objective_values = np.array(
        [result.objective for result in results],
        dtype=float,
    )

    execution_times = np.array(
        [result.elapsed_time for result in results],
        dtype=float,
    )

    feasibility_errors = np.array(
        [result.feasibility_error for result in results],
        dtype=float,
    )

    function_evaluations = np.array(
        [result.function_evaluations for result in results],
        dtype=float,
    )

    gaps_to_cc = np.array(
        [
            calculate_gap(value, charnes_cooper_value)
            for value in objective_values
        ]
    )

    gaps_to_dinkelbach = np.array(
        [
            calculate_gap(value, dinkelbach_value)
            for value in objective_values
        ]
    )

    return {
        "Method": method_name,
        "Best Z": float(objective_values.min()),
        "Mean Z": float(objective_values.mean()),
        "Worst Z": float(objective_values.max()),
        "SD": (
            float(objective_values.std(ddof=1))
            if len(objective_values) > 1
            else 0.0
        ),
        "Best Gap to Charnes-Cooper (%)": float(
            gaps_to_cc.min()
        ),
        "Mean Gap to Charnes-Cooper (%)": float(
            gaps_to_cc.mean()
        ),
        "Best Gap to Dinkelbach (%)": float(
            gaps_to_dinkelbach.min()
        ),
        "Mean Gap to Dinkelbach (%)": float(
            gaps_to_dinkelbach.mean()
        ),
        "Mean Time (s)": float(
            execution_times.mean()
        ),
        "Mean Function Evaluations": float(
            function_evaluations.mean()
        ),
        "Max Feasibility Error": float(
            feasibility_errors.max()
        ),
    }


def safe_file_name(method_name: str) -> str:
    return (
        method_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_matrix(result: Result) -> None:
    filename = (
        OUTPUT_DIRECTORY
        / f"solution_{safe_file_name(result.method)}.csv"
    )

    dataframe = pd.DataFrame(
        result.X,
        index=[f"Source_{i + 1}" for i in range(m)],
        columns=[f"Destination_{j + 1}" for j in range(n)],
    )

    dataframe.to_csv(filename)


def print_result(result: Result) -> None:
    positive_shipments = int(
        np.count_nonzero(
            result.X > FEASIBILITY_TOLERANCE
        )
    )

    print("\n" + "=" * 85)
    print(result.method)
    print("=" * 85)

    print(f"Objective value      : {result.objective:.12f}")
    print(f"Numerator            : {result.numerator:.6f}")
    print(f"Denominator          : {result.denominator:.6f}")
    print(f"Execution time       : {result.elapsed_time:.3f} seconds")
    print(f"Feasibility error    : {result.feasibility_error:.6e}")
    print(f"Feasible             : {is_feasible(result.X)}")
    print(f"Positive shipments   : {positive_shipments:,} of {dimension:,}")
    print(f"Function evaluations : {result.function_evaluations:,}")

    preview = pd.DataFrame(
        result.X[:10, :10],
        index=[f"S{i + 1}" for i in range(10)],
        columns=[f"D{j + 1}" for j in range(10)],
    )

    print("\nFirst 10 × 10 part of X:")
    print(preview.round(3).to_string())


def save_problem_data() -> None:
    pd.DataFrame(C).to_csv(
        OUTPUT_DIRECTORY / "numerator_coefficients_C.csv",
        index=False,
    )

    pd.DataFrame(D).to_csv(
        OUTPUT_DIRECTORY / "denominator_coefficients_D.csv",
        index=False,
    )

    pd.DataFrame({"Supply": supply}).to_csv(
        OUTPUT_DIRECTORY / "supply.csv",
        index=False,
    )

    pd.DataFrame({"Demand": demand}).to_csv(
        OUTPUT_DIRECTORY / "demand.csv",
        index=False,
    )


def draw_convergence_plot(
    charnes_cooper_result: Result,
    dinkelbach_result: Result,
    ga_best: Result,
    pso_best: Result,
    de_best: Result,
    gwo_best: Result,
) -> None:

    fig, ax = plt.subplots(figsize=(14, 8))

    ga_x = np.arange(1, len(ga_best.convergence) + 1)
    pso_x = np.arange(1, len(pso_best.convergence) + 1)
    de_x = np.arange(1, len(de_best.convergence) + 1)
    gwo_x = np.arange(1, len(gwo_best.convergence) + 1)
    dinkelbach_x = np.arange(1, len(dinkelbach_result.convergence) + 1)

    ax.plot(
        ga_x,
        ga_best.convergence,
        linestyle="-",
        marker="o",
        markevery=max(1, len(ga_x) // 10),
        linewidth=2.2,
        markersize=5,
        label="Genetic Algorithm (GA)",
    )

    ax.plot(
        pso_x,
        pso_best.convergence,
        linestyle="--",
        marker="s",
        markevery=max(1, len(pso_x) // 10),
        linewidth=2.2,
        markersize=5,
        label="Particle Swarm Optimization (PSO)",
    )

    ax.plot(
        de_x,
        de_best.convergence,
        linestyle="-.",
        marker="^",
        markevery=max(1, len(de_x) // 10),
        linewidth=2.6,
        markersize=6,
        label="Differential Evolution (DE)",
    )

    ax.plot(
        gwo_x,
        gwo_best.convergence,
        linestyle=":",
        marker="v",
        markevery=max(1, len(gwo_x) // 10),
        linewidth=2.6,
        markersize=6,
        label="Grey Wolf Optimizer (GWO)",
    )

    ax.axhline(
        y=charnes_cooper_result.objective,
        linestyle=(0, (8, 4)),
        linewidth=2.6,
        label="Charnes-Cooper",
    )

    ax.plot(
        dinkelbach_x,
        dinkelbach_result.convergence,
        linestyle=(0, (3, 2)),
        marker="D",
        linewidth=2.4,
        markersize=6,
        label="Dinkelbach",
    )

    all_values = (
        ga_best.convergence
        + pso_best.convergence
        + de_best.convergence
        + gwo_best.convergence
        + dinkelbach_result.convergence
        + [charnes_cooper_result.objective]
    )

    y_min = min(all_values)
    y_max = max(all_values)
    margin = max(0.01, 0.08 * (y_max - y_min))

    ax.set_xlim(
        1,
        max(
            len(ga_x),
            len(pso_x),
            len(de_x),
            len(gwo_x),
            len(dinkelbach_x),
        ),
    )

    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Objective Value")
    ax.set_title("Convergence of the Six Methods for the 100 × 100 LFTP")

    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 0.80, 1])

    fig.savefig(
        OUTPUT_DIRECTORY / "lftp_100x100_convergence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def draw_runtime_plot(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))

    plt.bar(
        summary["Method"],
        summary["Mean Time (s)"],
    )

    plt.xlabel("Method")
    plt.ylabel("Mean Execution Time in Seconds")
    plt.title("Execution-Time Comparison for the 100 × 100 LFTP")

    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY / "lftp_100x100_runtime.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def draw_final_objective_plot(
    best_results: list[Result],
) -> None:

    method_names = [result.method for result in best_results]
    objective_values = [result.objective for result in best_results]

    plt.figure(figsize=(12, 7))

    bars = plt.bar(
        method_names,
        objective_values,
    )

    plt.xlabel("Method")
    plt.ylabel("Final Objective Value")
    plt.title("Final Objective Values of the Six Methods for the 100 × 100 LFTP")

    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, objective_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY / "lftp_100x100_final_objectives.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def main() -> None:
    print("=" * 90)
    print("100 × 100 LINEAR FRACTIONAL TRANSPORTATION PROBLEM")
    print("=" * 90)

    print(f"Sources                  : {m}")
    print(f"Destinations             : {n}")
    print(f"Decision variables       : {dimension:,}")
    print(f"Total supply             : {supply.sum():,.0f}")
    print(f"Total demand             : {demand.sum():,.0f}")
    print(f"Balanced                 : {np.isclose(supply.sum(), demand.sum())}")
    print(f"Population size          : {POPULATION_SIZE}")
    print(f"Iterations               : {MAX_ITERATIONS}")
    print(f"Independent runs         : {INDEPENDENT_RUNS}")
    print(f"Output directory         : {OUTPUT_DIRECTORY}")

    if not np.isclose(
        supply.sum(),
        demand.sum(),
    ):
        raise RuntimeError("The generated problem is not balanced.")

    save_problem_data()

    print("\nRunning Charnes-Cooper...")
    charnes_cooper_result = solve_charnes_cooper()

    print(
        f"Charnes-Cooper: Z = {charnes_cooper_result.objective:.12f}, "
        f"time = {charnes_cooper_result.elapsed_time:.3f} seconds"
    )

    print("\nRunning Dinkelbach...")
    dinkelbach_result = solve_dinkelbach()

    print(
        f"Dinkelbach: Z = {dinkelbach_result.objective:.12f}, "
        f"time = {dinkelbach_result.elapsed_time:.3f} seconds"
    )

    exact_difference = abs(
        charnes_cooper_result.objective
        - dinkelbach_result.objective
    )

    exact_relative_difference = (
        exact_difference
        / max(
            abs(charnes_cooper_result.objective),
            abs(dinkelbach_result.objective),
            1e-15,
        )
    )

    print("\n" + "=" * 80)
    print("CHARNES-COOPER AND DINKELBACH")
    print("=" * 80)
    print(f"Charnes-Cooper objective : {charnes_cooper_result.objective:.12f}")
    print(f"Dinkelbach objective     : {dinkelbach_result.objective:.12f}")
    print(f"Absolute difference      : {exact_difference:.12e}")
    print(f"Relative difference      : {exact_relative_difference:.12e}")

    if exact_relative_difference > 1e-7:
        raise RuntimeError("Charnes-Cooper and Dinkelbach do not agree.")

    ga_best, ga_runs = run_repeated(
        solve_genetic_algorithm,
        "Genetic Algorithm",
        100,
    )

    pso_best, pso_runs = run_repeated(
        solve_particle_swarm,
        "Particle Swarm Optimization",
        200,
    )

    de_best, de_runs = run_repeated(
        solve_differential_evolution,
        "Differential Evolution",
        300,
    )

    gwo_best, gwo_runs = run_repeated(
        solve_grey_wolf,
        "Grey Wolf Optimizer",
        400,
    )

    best_results = [
        charnes_cooper_result,
        dinkelbach_result,
        ga_best,
        pso_best,
        de_best,
        gwo_best,
    ]

    for result in best_results:
        print_result(result)
        save_matrix(result)

    cc_value = charnes_cooper_result.objective
    dinkelbach_value = dinkelbach_result.objective

    cc_gap_to_dinkelbach = calculate_gap(
        cc_value,
        dinkelbach_value,
    )

    dinkelbach_gap_to_cc = calculate_gap(
        dinkelbach_value,
        cc_value,
    )

    summary_rows = [
        {
            "Method": "Charnes-Cooper",
            "Best Z": cc_value,
            "Mean Z": cc_value,
            "Worst Z": cc_value,
            "SD": 0.0,
            "Best Gap to Charnes-Cooper (%)": 0.0,
            "Mean Gap to Charnes-Cooper (%)": 0.0,
            "Best Gap to Dinkelbach (%)": cc_gap_to_dinkelbach,
            "Mean Gap to Dinkelbach (%)": cc_gap_to_dinkelbach,
            "Mean Time (s)": charnes_cooper_result.elapsed_time,
            "Mean Function Evaluations": charnes_cooper_result.function_evaluations,
            "Max Feasibility Error": charnes_cooper_result.feasibility_error,
        },
        {
            "Method": "Dinkelbach",
            "Best Z": dinkelbach_value,
            "Mean Z": dinkelbach_value,
            "Worst Z": dinkelbach_value,
            "SD": 0.0,
            "Best Gap to Charnes-Cooper (%)": dinkelbach_gap_to_cc,
            "Mean Gap to Charnes-Cooper (%)": dinkelbach_gap_to_cc,
            "Best Gap to Dinkelbach (%)": 0.0,
            "Mean Gap to Dinkelbach (%)": 0.0,
            "Mean Time (s)": dinkelbach_result.elapsed_time,
            "Mean Function Evaluations": dinkelbach_result.function_evaluations,
            "Max Feasibility Error": dinkelbach_result.feasibility_error,
        },
        summarize_stochastic_results(
            "Genetic Algorithm",
            ga_runs,
            cc_value,
            dinkelbach_value,
        ),
        summarize_stochastic_results(
            "Particle Swarm Optimization",
            pso_runs,
            cc_value,
            dinkelbach_value,
        ),
        summarize_stochastic_results(
            "Differential Evolution",
            de_runs,
            cc_value,
            dinkelbach_value,
        ),
        summarize_stochastic_results(
            "Grey Wolf Optimizer",
            gwo_runs,
            cc_value,
            dinkelbach_value,
        ),
    ]

    summary = pd.DataFrame(summary_rows)

    comparison_file = OUTPUT_DIRECTORY / "lftp_100x100_comparison.csv"

    summary.to_csv(
        comparison_file,
        index=False,
    )

    print("\n" + "=" * 170)
    print("FINAL COMPARISON")
    print("=" * 170)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    draw_convergence_plot(
        charnes_cooper_result,
        dinkelbach_result,
        ga_best,
        pso_best,
        de_best,
        gwo_best,
    )

    draw_runtime_plot(summary)
    draw_final_objective_plot(best_results)

    print("\nAll results were saved in:")
    print(OUTPUT_DIRECTORY.resolve())

if __name__ == "__main__":
    main()