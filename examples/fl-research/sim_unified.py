"""Unified FL experiment runner: same client, data, rounds and aggregation metrics.
Run from examples/fl-research with repository's Flower environment activated.
"""
import os
import random
import numpy as np
import flwr as fl

from client_unified import client_fn
from fedavg_with_logging import FedAvgWithLogging
from OORT_with_logging import OORTWithClientLogging
from proposed_with_logging import ProposedWithLogging
from sim_fedavg import aggregate_fit_metrics, aggregate_eval_metrics

STRATEGY = os.environ.get('STRATEGY', 'fedavg').strip().lower()
ALIASES = {'proposed_score': 'proposed_score', 'proposed_random': 'proposed_random',
           'proposed_oort': 'proposed_oort', 'oort': 'oort', 'fedavg': 'fedavg'}
if STRATEGY not in ALIASES:
    raise ValueError(f'Unknown STRATEGY={STRATEGY!r}; choose {list(ALIASES)}')

NUM_CLIENTS = int(os.getenv('NUM_CLIENTS', '20'))
SELECTED_CLIENTS = int(os.getenv('SELECTED_CLIENTS', '5'))
SHORTLIST_SIZE = int(os.getenv('SHORTLIST_SIZE', '10'))
ROUNDS = int(os.getenv('ROUNDS', '3'))
LOCAL_EPOCHS = int(os.getenv('LOCAL_EPOCHS', '1'))
SEED = int(os.getenv('SELECTION_SEED', '1234'))
TARGET_ACCURACY = float(os.getenv('TARGET_ACCURACY', '0.70'))
LOG_DIR = os.getenv('LOG_DIR', f'results/unified_{STRATEGY}_seed{SEED}')

if not (1 <= SELECTED_CLIENTS <= NUM_CLIENTS):
    raise ValueError('Expected 1 <= SELECTED_CLIENTS <= NUM_CLIENTS')
if STRATEGY.startswith('proposed') and not (SELECTED_CLIENTS <= SHORTLIST_SIZE <= NUM_CLIENTS):
    raise ValueError('Expected SELECTED_CLIENTS <= SHORTLIST_SIZE <= NUM_CLIENTS')

random.seed(SEED)
np.random.seed(SEED)

def fit_config(round_number):
    return {'server_round': int(round_number), 'local_epochs': LOCAL_EPOCHS}

shared = dict(
    num_clients=NUM_CLIENTS, log_dir=LOG_DIR, target_accuracy=TARGET_ACCURACY,
    fraction_fit=SELECTED_CLIENTS / NUM_CLIENTS,
    min_fit_clients=SELECTED_CLIENTS, min_available_clients=NUM_CLIENTS,
    fraction_evaluate=SELECTED_CLIENTS / NUM_CLIENTS,
    min_evaluate_clients=SELECTED_CLIENTS,
    on_fit_config_fn=fit_config,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
)

if STRATEGY == 'fedavg':
    strategy = FedAvgWithLogging(**shared)
elif STRATEGY == 'oort':
    strategy = OORTWithClientLogging(
        **shared, seed=SEED,
        exploration_factor=float(os.getenv('OORT_EXPLORATION_FACTOR', '0.90')),
        exploration_decay=float(os.getenv('OORT_EXPLORATION_DECAY', '0.98')),
        min_exploration_factor=float(os.getenv('OORT_MIN_EXPLORATION_FACTOR', '0.20')),
        pacer_window=int(os.getenv('OORT_PACER_WINDOW', '20')),
        pacer_step=float(os.getenv('OORT_PACER_STEP', '60.0')),
        straggler_penalty_alpha=float(os.getenv('OORT_ALPHA', '2.0')),
        cutoff_percentage=float(os.getenv('OORT_CUTOFF', '0.95')),
        max_selection_per_client=int(os.getenv('OORT_MAX_SELECTIONS', '10')),
    )
else:
    final_selector = {
        'proposed_score': 'combined_score',
        'proposed_random': 'fedavg',
        'proposed_oort': 'oort',
    }[STRATEGY]
    strategy = ProposedWithLogging(
        **shared, selected_clients=SELECTED_CLIENTS,
        shortlist_size=SHORTLIST_SIZE,
        selection_mode=os.getenv('SELECTION_MODE', 'score'),
        final_selector=final_selector,
        capability_weight=float(os.getenv('CAPABILITY_WEIGHT', '0.5')),
        selection_seed=SEED,
    )

if __name__ == '__main__':
    print(f'STRATEGY={STRATEGY} NUM_CLIENTS={NUM_CLIENTS} '
          f'SELECTED_CLIENTS={SELECTED_CLIENTS} SHORTLIST_SIZE={SHORTLIST_SIZE} '
          f'ROUNDS={ROUNDS} LOCAL_EPOCHS={LOCAL_EPOCHS} SEED={SEED} LOG_DIR={LOG_DIR}', flush=True)
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS), strategy=strategy,
        client_resources={
            'num_cpus': float(os.getenv('CLIENT_CPUS', '4')),
            'num_gpus': float(os.getenv('CLIENT_GPUS', '1.0')),
        },
    )
