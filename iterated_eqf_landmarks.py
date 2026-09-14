#!/usr/bin/env python3
# Copyright (C) 2026 Pieter van Goor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Monte Carlo comparison of one-step and iterated EqFs on SE(2).

The vehicle follows a tangent-aligned figure-eight Lissajous curve. By default
it uses two range landmarks; pass --third-landmark to add a third. Outputs:
trajectory/error plots, average NEES with chi-square bounds.
"""

import argparse
from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

plt.rc('lines', linewidth=1.0)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

cross_matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
FILTERS = ("Standard EqF", "Iterated EqF")


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def se2_exp(xi):
    th, rho = float(xi[0]), np.asarray(xi[1:3], dtype=float)
    if abs(th) < 1e-8:
        V = np.eye(2) + 0.5 * th * cross_matrix - th**2 / 6.0 * np.eye(2)
    else:
        V = np.sin(th) / th * np.eye(2) + (1.0 - np.cos(th)) / th * cross_matrix
    X = np.eye(3)
    X[:2, :2], X[:2, 2] = rot(th), V @ rho
    return X


def se2_log(X):
    th = np.arctan2(X[1, 0], X[0, 0])
    if abs(th) < 1e-8:
        Vinv = np.eye(2) - 0.5 * th * cross_matrix - th**2 / 12.0 * np.eye(2)
    else:
        Vinv = 0.5 * th / np.tan(0.5 * th) * np.eye(2) - 0.5 * th * cross_matrix
    return np.r_[th, Vinv @ X[:2, 2]]


def inv_se2(X):
    Y = np.eye(3)
    Y[:2, :2] = X[:2, :2].T
    Y[:2, 2] = -Y[:2, :2] @ X[:2, 2]
    return Y

def se2_ad(eps):
    return np.array([
        [0,0,0],
        [eps[2], 0, -eps[0]],
        [-eps[1], eps[0], 0]
    ])

def epsilon_jacobian(eps):
    t = eps[0]
    if abs(t) > 1e-8:
        beta = t**-2 - (1+np.cos(t)) / (2*t*np.sin(t))
    else:
        beta = (1.0 / 12.0 + t**2 / 720.0 + t**4 / 30240.0)
    ad = se2_ad(eps)
    J = np.eye(3) - 0.5 * ad + beta * ad@ad
    return J

def pose(theta, p):
    X = np.eye(3)
    X[:2, :2], X[:2, 2] = rot(theta), p
    return X


def state(X):
    return np.arctan2(X[1, 0], X[0, 0]), X[:2, 2].copy()


def solve_spd(A, b):
    A = 0.5 * (A + A.T)
    try:
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, b))
    except np.linalg.LinAlgError:
        return np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), b)


def make_landmarks(use_third=False):
    landmarks = [[-7, 5], [-7, -5]]
    if use_third:
        landmarks.append([4, 0])
    return np.asarray(landmarks, dtype=float)


def ranges(X, landmarks):
    return np.linalg.norm(X[:2, 2][None, :] - landmarks, axis=1)


def range_jacobian_left(X, landmarks):
    p = X[:2, 2]
    d = p[None, :] - landmarks
    q = d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)
    return np.column_stack((q @ (cross_matrix @ p), q))


def map_cost(X, Xprior, Pprior, y, landmarks, R):
    e = se2_log(X @ inv_se2(Xprior))
    r = y - ranges(X, landmarks)
    return 0.5 * (e @ solve_spd(Pprior, e) + r @ solve_spd(R, r))


def update_one_shot(Xprior, Pprior, y, landmarks, R):
    C = range_jacobian_left(Xprior, landmarks)
    innovation = y - ranges(Xprior, landmarks)
    S = C @ Pprior @ C.T + R
    K = Pprior @ C.T @ np.linalg.inv(S)
    Xpost = se2_exp(K @ innovation) @ Xprior
    Ppost = (np.eye(3) - K @ C) @ Pprior
    return Xpost, 0.5 * (Ppost + Ppost.T), 1


def update_iterated(Xprior, Pprior, y, landmarks, R, max_iter=12, tol=1e-8):
    X = Xprior.copy()
    old_cost = map_cost(X, Xprior, Pprior, y, landmarks, R)

    for iteration in range(1, max_iter + 1):
        eps = se2_log(X @ inv_se2(Xprior))
        J = epsilon_jacobian(eps)
        C = range_jacobian_left(X, landmarks)
        innovation = y - ranges(X, landmarks)
        JInv = np.linalg.inv(J)
        PCheck = JInv @ Pprior @ JInv.T

        S = C @ PCheck @ C.T + R
        K = PCheck @ C.T @ np.linalg.inv(S)
        Delta = K @ (innovation + C @ JInv @ eps) - JInv @ eps
        
        alpha = 1.0
        while alpha > 1.0 / 64.0:
            Xtrial = se2_exp(alpha * Delta) @ X
            trial_cost = map_cost(Xtrial, Xprior, Pprior, y, landmarks, R)
            if trial_cost <= old_cost + 1e-12:
                break
            alpha *= 0.5
        X, old_cost = Xtrial, trial_cost
        if np.linalg.norm(alpha * Delta) < tol:
            break

    Ppost = (np.eye(3) - K @ C) @ PCheck
    return X, 0.5 * (Ppost + Ppost.T), iteration

@dataclass
class Config:
    dt: float = 0.02
    duration: float = 80.0
    measurement_dt: float = 0.5
    sigma0_angle: float = np.pi/2
    sigma0_position: float = 2.0
    sigma_range: float = 0.1
    sigma_omega: float = 0.01
    sigma_speed: float = 0.01
    amplitude_x: float = 6.0
    amplitude_y: float = 4.0
    period: float = 40.0
    seed: int = 1
    use_third_landmark: bool = False
    show_nees: bool = False


def truth(t_now, cfg):
    w = 2.0 * np.pi / cfg.period
    p = np.array([cfg.amplitude_x * np.sin(w * t_now),
                  cfg.amplitude_y * np.sin(2.0 * w * t_now)])
    dp = np.array([cfg.amplitude_x * w * np.cos(w * t_now),
                   2.0 * cfg.amplitude_y * w * np.cos(2.0 * w * t_now)])
    ddp = np.array([-cfg.amplitude_x * w**2 * np.sin(w * t_now),
                    -4.0 * cfg.amplitude_y * w**2 * np.sin(2.0 * w * t_now)])
    speed = np.linalg.norm(dp)
    theta = np.arctan2(dp[1], dp[0])
    omega = (dp[0] * ddp[1] - dp[1] * ddp[0]) / speed**2
    return pose(theta, p), np.array([omega, speed])


def B_matrix(X):
    B = np.zeros((3, 2))
    B[0, 0] = 1.0
    B[1:, 0] = -cross_matrix @ X[:2, 2]
    B[1:, 1] = X[:2, 0]
    return B


def propagate(X, P, u_measured, Qc, dt):
    X = X @ se2_exp(np.array([u_measured[0] * dt,
                              u_measured[1] * dt, 0.0]))
    B = B_matrix(X)
    P = P + B @ Qc @ B.T * dt
    return X, 0.5 * (P + P.T)


def run_once(cfg):
    rng = np.random.default_rng(cfg.seed)
    landmarks = make_landmarks(cfg.use_third_landmark)
    R = cfg.sigma_range**2 * np.eye(len(landmarks))
    Qc = np.diag([cfg.sigma_omega**2, cfg.sigma_speed**2])

    Xtrue0, _ = truth(0.0, cfg)
    

    P0 = np.diag([cfg.sigma0_angle**2, cfg.sigma0_position**2, cfg.sigma0_position**2])
    eps0 = rng.multivariate_normal(np.zeros(3), P0)
    X0 = se2_exp(eps0) @ Xtrue0
    filters = {name: [X0.copy(), P0.copy()] for name in FILTERS}

    count = int(round(cfg.duration / cfg.dt)) + 1
    stride = max(1, int(round(cfg.measurement_dt / cfg.dt)))
    times = np.arange(count) * cfg.dt
    true_states = np.zeros((count, 3))
    estimates = {name: np.zeros((count, 3)) for name in FILTERS}
    covariances = {name: np.zeros((count, 3, 3)) for name in FILTERS}
    iterations = []

    for k, t_now in enumerate(times):
        Xtrue, u = truth(t_now, cfg)
        theta, p = state(Xtrue)
        true_states[k] = [theta, *p]
        for name, (X, P) in filters.items():
            theta_hat, p_hat = state(X)
            estimates[name][k] = [theta_hat, *p_hat]
            covariances[name][k] = P
        if k == count - 1:
            break

        u_measured = u + rng.multivariate_normal(np.zeros(2), Qc / cfg.dt)
        for name in FILTERS:
            filters[name][0], filters[name][1] = propagate(
                filters[name][0], filters[name][1], u_measured, Qc, cfg.dt)

        if (k + 1) % stride == 0:
            Xtrue_next, _ = truth((k + 1) * cfg.dt, cfg)
            y = ranges(Xtrue_next, landmarks) + rng.normal(
                0.0, cfg.sigma_range, len(landmarks))
            X, P = filters[FILTERS[0]]
            X, P, _ = update_one_shot(X, P, y, landmarks, R)
            filters[FILTERS[0]] = [X, P]
            X, P = filters[FILTERS[1]]
            X, P, nit = update_iterated(X, P, y, landmarks, R)
            filters[FILTERS[1]] = [X, P]
            iterations.append(nit)

    result = {"times": times, "true": true_states, "estimates": estimates,
              "covariances": covariances, "landmarks": landmarks,
              "iterations": np.asarray(iterations)}
    for name in FILTERS:
        pos_error = np.linalg.norm(estimates[name][:, 1:3] - true_states[:, 1:3], axis=1)
        ang_error = np.abs(wrap(estimates[name][:, 0] - true_states[:, 0]))
        nees = np.empty(count)
        for k in range(count):
            Xt = pose(true_states[k, 0], true_states[k, 1:3])
            Xh = pose(estimates[name][k, 0], estimates[name][k, 1:3])
            error = se2_log(Xt @ inv_se2(Xh))
            nees[k] = error @ solve_spd(covariances[name][k], error)
        result[name] = {"pos_error": pos_error,
                        "ang_error": ang_error,
                        "nees": nees}
    return result


def mean_ci(samples, axis=0, confidence=0.95):
    samples = np.asarray(samples)
    mean = np.mean(samples, axis=axis)
    upper = np.percentile(samples, 2.5, axis=axis)
    lower = np.percentile(samples, 97.5, axis=axis)
    return mean, upper, lower


def monte_carlo(cfg, runs):
    trials = [run_once(replace(cfg, seed=cfg.seed + i)) for i in range(runs)]
    summary = {"runs": runs, "times": trials[0]["times"],
               "landmark_count": len(trials[0]["landmarks"])}
    # Iteration-count distribution at every measurement update.
    iteration_samples = np.stack([trial["iterations"] for trial in trials])
    update_times = (
        np.arange(1, iteration_samples.shape[1] + 1)
        * cfg.measurement_dt
    )
    summary["iteration_statistics"] = {
        "times": update_times,
        "mean": np.mean(iteration_samples, axis=0),
        "median": np.median(iteration_samples, axis=0),
        "lower": np.percentile(iteration_samples, 2.5, axis=0),
        "upper": np.percentile(iteration_samples, 97.5, axis=0),
    }

    for name in FILTERS:
        pe = np.stack([trial[name]["pos_error"] for trial in trials])
        ae = np.stack([trial[name]["ang_error"] for trial in trials])
        nees = np.stack([trial[name]["nees"] for trial in trials])
        pos_rmse = np.sqrt(np.mean(pe**2, axis=1))
        heading_rmse = np.rad2deg(np.sqrt(np.mean(ae**2, axis=1)))
        summary[name] = {
            "pos_curve": mean_ci(pe),
            "heading_curve": mean_ci(np.rad2deg(ae)),
            "average_nees": np.mean(nees, axis=0),
            "pos_rmse_samples": pos_rmse,
            "heading_rmse_samples": heading_rmse,
            "pos_rmse": mean_ci(pos_rmse),
            "heading_rmse": mean_ci(heading_rmse),
        }
    summary["iterations"] = mean_ci(
        np.array([np.mean(trial["iterations"]) for trial in trials]))
    summary["anees_bounds"] = (
        chi2.ppf(0.025, 3 * runs) / runs,
        chi2.ppf(0.975, 3 * runs) / runs,
    )
    return trials[0], summary


def print_summary(summary):
    print(f"Monte Carlo: {summary['runs']} runs, {summary['landmark_count']} landmarks")
    print("Mean run-level RMSE with 95% empirical percentile bounds")
    for name in FILTERS:
        pm, pl, pu = summary[name]["pos_rmse"]
        hm, hl, hu = summary[name]["heading_rmse"]
        print(f"{name:20s} position RMSE = {pm:.3f} m [{pl:.3f}, {pu:.3f}], "
              f"heading RMSE = {hm:.2f} deg [{hl:.2f}, {hu:.2f}]")
    m, lo, hi = summary["iterations"]
    print(f"Iterated EqF iterations/update = {m:.2f} [{lo:.2f}, {hi:.2f}]")

    lower, upper = summary["anees_bounds"]
    print(f"\nAverage-NEES 95% bounds: [{lower:.3f}, {upper:.3f}], expected value 3")
    for name in FILTERS:
        nees = summary[name]["average_nees"]
        fraction = np.mean((nees >= lower) & (nees <= upper))
        print(f"{name:20s} time points inside bounds = {100.0*fraction:.1f}%")


def plot_results(example, summary, show_nees):
    times, true = example["times"], example["true"]
    colors = {FILTERS[0]: "C0", FILTERS[1]: "C3"}
    nrows = 5 if show_nees else 4
    fig, axes = plt.subplots(
        nrows, 1, figsize=(6, 1.75 * nrows), constrained_layout=True
    )

    axes[0].plot(true[:, 1], true[:, 2], "k", label="True", ls="--")
    for name in FILTERS:
        estimate = example["estimates"][name]
        axes[0].plot(
            estimate[:, 1], estimate[:, 2], color=colors[name], label=name
        )
    landmarks = example["landmarks"]
    axes[0].scatter(
        landmarks[:, 0], landmarks[:, 1], s=100,
        marker="*", color="C2", label="Landmarks"
    )
    axes[0].set(
        title=f"Example {len(landmarks)}-Landmark Lissajous Trajectory",
        xlabel="Position x [m]", ylabel="Position y [m]"
    )
    axes[0].set_xlim([-10, 20])
    axes[0].grid(True)
    axes[0].legend(loc="right")

    for name in FILTERS:
        mean, lower, upper = summary[name]["pos_curve"]
        axes[1].plot(times, mean, color=colors[name], label=name)
        axes[1].fill_between(times, lower, upper, color=colors[name], alpha=0.2)
        mean, lower, upper = summary[name]["heading_curve"]
        axes[2].plot(times, mean, color=colors[name], label=name)
        axes[2].fill_between(times, lower, upper, color=colors[name], alpha=0.2)
    axes[1].set(
        title=r"Position Error Mean and 95\% Bounds",
        xlabel="Time [s]", ylabel="position error [m]", xlim=[0, times[-1]]
    )
    axes[2].set(
        title=r"Heading Error Mean and 95\% Bounds",
        xlabel="Time [s]", ylabel="heading error [deg]", xlim=[0, times[-1]]
    )
    for axis in axes[1:3]:
        axis.grid(True)
        axis.set_yscale("log")
    axes[1].legend()

    stats = summary["iteration_statistics"]
    update_times = stats["times"]
    axes[3].step(
        update_times, stats["mean"], where="mid",
        color=colors[FILTERS[1]], label="Mean"
    )
    axes[3].fill_between(
        update_times, stats["lower"], stats["upper"], step="mid",
        color=colors[FILTERS[1]], alpha=0.2,
        label=r"95\% bounds"
    )
    axes[3].set(
        title="Iterations per Measurement Update",
        xlabel="Time [s]", ylabel=r"\# Iterations", xlim=[0, times[-1]]
    )
    axes[3].set_ylim(bottom=0.5)
    axes[3].grid(True)
    axes[3].legend(loc="upper right")

    if show_nees:
        for name in FILTERS:
            axes[4].plot(
                times, summary[name]["average_nees"],
                color=colors[name], label=name
            )
        lower, upper = summary["anees_bounds"]
        axes[4].axhspan(
            lower, upper, color="0.75", alpha=0.5,
            label=r"95\% consistency region"
        )
        axes[4].axhline(
            3.0, color="k", ls="--", label="Expected NEES = 3"
        )
        axes[4].set(
            title=f"Mean NEES Over N={summary['runs']} Trials",
            xlabel="Time [s]", ylabel="NEES", xlim=[0, times[-1]]
        )
        axes[4].grid(True)
        axes[4].legend(loc="upper right",ncols=2)
        axes[4].set_yscale("log")
        axes[4].set_ylim([1,20])

    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mc", type=int, default=100,
                        help="number of Monte Carlo runs")
    parser.add_argument("--third-landmark", action="store_true",
                        help="add landmark [4, 0] to the default two")
    parser.add_argument("--save", default="iterated_eqf_statistics.pdf")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--nees", action="store_true")
    args = parser.parse_args()
    if args.mc < 2:
        parser.error("--mc must be at least 2")

    cfg = Config(seed=args.seed, use_third_landmark=args.third_landmark, show_nees=args.nees)
    example, summary = monte_carlo(cfg, args.mc)
    print_summary(summary)
    figure = plot_results(example, summary, cfg.show_nees)
    figure.savefig(args.save, dpi=180)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
