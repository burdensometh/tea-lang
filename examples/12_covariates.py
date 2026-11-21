"""Temporal covariate join — attach participant history to event rows.

An event-level table records one row per observation with a timestamp; a
separate history table records participant attributes that take effect at some
time and may later expire.  ``attach_covariates`` attaches, for each event, the
single history record that was in force at that event's time.

Expected outcome: Mann-Whitney U / Welch's T / Kruskal-Wallis on the attached
``group`` attribute; the sample size is unchanged by the join.

Run:  poetry run python examples/12_covariates.py
"""
import os

import pandas as pd

import tea

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data')

events = pd.read_csv(os.path.join(DATA_DIR, 'covariate_events.csv'))
history = pd.read_csv(os.path.join(DATA_DIR, 'covariate_history.csv'))

tea.data(events, key='pid')
tea.define_variables([
    {'name': 'group', 'data type': 'nominal', 'categories': ['control', 'treatment']},
    {'name': 'outcome', 'data type': 'ratio'},
])
tea.define_study_design({
    'study type': 'experiment',
    'independent variables': 'group',
    'dependent variables': 'outcome',
})
tea.assume({'Type I (False Positive) Error Rate': 0.05})

# Declare how the history attaches to the events.
tea.attach_covariates(
    history,
    tea.CovariateJoinSpec(
        event_time='event_date',
        effective_time='effective_date',
        expires_time='expires_date',
        participant_key='pid',
        on_overlap='latest_effective',
        on_duplicate='error',
        on_missing='error',
    ),
)

results = tea.hypothesize(['group', 'outcome'], ['group:control < treatment'])

evidence = tea.get_covariate_evidence()
print(evidence)
