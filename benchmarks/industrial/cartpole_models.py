"""Small learned state-space controls using native scikit-learn and SciPy."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brute
from sklearn.linear_model import BayesianRidge


class DeltaModel:
    def __init__(self, regressors, *, action_conditioned):
        self.regressors = regressors
        self.action_conditioned = action_conditioned
        self.a = np.eye(4) + np.stack([r.coef_[:4] for r in regressors])
        self.b = np.array([r.coef_[-1] for r in regressors]) if action_conditioned else np.zeros(4)
        self.intercept = np.array([r.intercept_ for r in regressors])

    @classmethod
    def fit(cls, states, actions, next_states, *, action_conditioned):
        x = np.asarray(states)
        if action_conditioned:
            x = np.column_stack([x, np.asarray(actions) * 2 - 1])
        delta = np.asarray(next_states) - np.asarray(states)
        return cls([BayesianRidge().fit(x, delta[:, i]) for i in range(4)],
                   action_conditioned=action_conditioned)

    def mean_step(self, state, action):
        return self.a @ state + self.b * (action * 2 - 1) + self.intercept

    def __call__(self, request):
        state = np.array(request['initial_state'], dtype=float)
        covariance = np.zeros((4, 4))
        states, deviations = [], []
        for action in request['actions']:
            features = np.append(state, action * 2 - 1) if self.action_conditioned else state
            predicted = [r.predict(features.reshape(1, -1), return_std=True) for r in self.regressors]
            innovations = np.array([p[1][0] ** 2 for p in predicted])
            native_mean = state + np.array([p[0][0] for p in predicted])
            # Exported coefficients must reproduce the upstream predictor used for forecasts.
            np.testing.assert_allclose(native_mean, self.mean_step(state, action), rtol=1e-10, atol=1e-10)
            state = native_mean
            covariance = self.a @ covariance @ self.a.T + np.diag(innovations)
            states.append(state.tolist())
            deviations.append(np.sqrt(np.diag(covariance)).tolist())
        return {'states': states, 'standard_deviation': deviations, 'metadata': {
            'uncertainty': 'Gaussian marginals; linear covariance propagation with independent innovations',
            'limits': 'Temporal parameter correlations and nonlinear model error are not represented'}}

    def parameters(self):
        return {'action_conditioned': self.action_conditioned,
                'a': self.a.tolist(), 'b': self.b.tolist(), 'intercept': self.intercept.tolist(),
                'regressors': [{'coef': r.coef_.tolist(), 'intercept': float(r.intercept_),
                                'sigma': r.sigma_.tolist(), 'alpha': float(r.alpha_),
                                'lambda': float(r.lambda_), 'parameters': r.get_params()}
                               for r in self.regressors]}


class Persistence:
    def mean_step(self, state, action):
        return np.array(state, copy=True)

    def __call__(self, request):
        return {'states': [list(request['initial_state']) for _ in request['actions']],
                'metadata': {'uncertainty': 'unavailable'}}


def plan(model, observation):
    """Native exhaustive search over the protocol's fixed finite action space."""
    costs = []
    def objective(actions):
        state = np.array(observation, dtype=float)
        cost = 0.0
        for action in actions:
            state = model.mean_step(state, action)
            cost += float(np.dot([1, 0.05, 20, 0.5], state ** 2)) + 0.01
        costs.append({'actions': [int(a) for a in actions], 'cost': cost})
        return cost
    actions, value, _, _ = brute(objective, ranges=(slice(0, 2, 1),) * 6,
                                 full_output=True, finish=None, workers=1)
    return int(actions[0]), {'observation': np.asarray(observation).tolist(),
                            'plan': [int(a) for a in actions], 'cost': float(value),
                            'candidate_costs': costs, 'evaluations': len(costs)}
