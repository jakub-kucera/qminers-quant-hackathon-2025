"""
Script to manually add trial 103 parameters to an Optuna study.
This allows you to seed a study with known good parameters.
"""

import optuna

# Trial 103 parameters
trial_103_params = {
    'ensamble_required_n': 700,
    'ensamble_retrain_freq': 200,
    'pythagorean_power': 1.4452228018354543,
    'pythagorean_regularization': 25,
    'pythagorean_decay': 0.9961543934996865,
    'four_factor_required_n': 400,
    'four_factor_decay': 0.9929968149932047,
    'gradient_descent_epochs': 17,
    'gradient_descent_min_games': 500,
    'exhaustion_b2b_penalty': 1.3835167068873355,
    'exhaustion_road_penalty': 0.36714393446512866,
    'elo_k_base': 30,
    'elo_home_advantage': 35,
    'elo_margin_cap': 2,
    'poisson_regularization': 3.459346261455932,
    'poisson_decay': 0.992628863991723,
    'rolling_window_short': 7,
    'rolling_window_med': 10,
    'rolling_window_long': 20,
    'pepe_min_edge': 0.05493318133185403,
    'pepe_kelly_fraction': 0.5287748436277171,
    'pepe_value_multiplier': 3.9642362123262553,
    'pepe_confidence_threshold': 0.6327858468816776,
    'pepe_max_bet_fraction': 0.11156382293203759,
    'pepe_max_total_exposure': 0.5940450771286425,
    'pepe_confidence_weight_separation': 0.6602736604741463,
    'pepe_confidence_weight_certainty': 0.16928670385859046
}

# Trial 103 value
trial_103_value = 1170.5300000000002

# Study configuration
study_name = 'hybrid_comprehensive_v2'
storage = 'sqlite:///hybrid_comprehensive_v2.db'

# Create or load study
study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction='maximize',
    load_if_exists=True
)

print(f"Study: {study_name}")
print(f"Current trials: {len(study.trials)}")

# Add trial 103 using enqueue_trial
study.enqueue_trial(trial_103_params)
print(f"\n✓ Enqueued trial 103 parameters")
print(f"  Value: ${trial_103_value:.2f}")
print(f"  This trial will be evaluated next when you run the optimization")

# Alternatively, you can tell the study about this trial's value
# by running one trial with these exact parameters
print("\nTo actually add this trial with its value, run:")
print(f"  cd src && uv run python add_trial_103.py --evaluate")
print("\nOr just continue optimization and it will be evaluated:")
print(f"  cd src && uv run optimize_hybrid_comprehensive.py --n-trials 100 --n-jobs 4 --study-name {study_name} --storage {storage}")
