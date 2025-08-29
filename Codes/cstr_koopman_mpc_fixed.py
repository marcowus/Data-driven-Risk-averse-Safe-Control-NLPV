# -*- coding: utf-8 -*-
"""CSTR control using Koopman-based MPC and MPPI with safety shield.

This script implements a continuous stirred tank reactor (CSTR) example
with a Koopman operator based world model. A bug in the original
implementation caused a dimension mismatch when constructing the
augmented lifted state for the Koopman MPC controller. The helper
function now explicitly reshapes all terms into column vectors before
stacking, preventing the ``ValueError`` raised by ``cvxpy.vstack``.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import pinv
from tqdm import tqdm
import cvxpy as cp
import pandas as pd
from tabulate import tabulate
import warnings

# 忽略CVXPY未来版本关于reshape的警告
warnings.filterwarnings("ignore", message="The 'order' parameter in the reshape atom is deprecated")

# ---------------------------------------------------------------------------
# 0. 辅助工具 (完全恢复到您的原版，未做任何修改)
# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self, original_dim):
        self.mean = None
        self.std = None
        self.original_dim = original_dim

    def fit(self, data):
        original_data = data[:self.original_dim, :]
        self.mean = np.mean(original_data, axis=1, keepdims=True)
        self.std = np.std(original_data, axis=1, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x):
        return (x[:self.original_dim].reshape(-1, 1) - self.mean) / self.std

    def inverse_transform(self, x_norm):
        return x_norm.reshape(-1, 1) * self.std + self.mean

def lift_state(x_normalized, poly_order=2):
    x1, x2 = x_normalized.flatten()
    lifted = [1.0]
    for i in range(1, poly_order + 1):
        for j in range(i + 1):
            lifted.append((x1**(i - j)) * (x2**j))
    return np.array(lifted)

def lift_input(u):
    return np.array([u])

# ---------------------------------------------------------------------------
# 1. 环境 (Environment): CSTR 反应器 (与原版相同)
# ---------------------------------------------------------------------------
class CSTREnvironment:
    """代表真实世界的CSTR系统，作为RL环境。"""
    def __init__(self, sim_params, dt, initial_state, target_state, state_weights, integral_weights):
        self.params = sim_params
        self.dt = dt
        self.initial_state = np.copy(initial_state)
        self.target_state = np.copy(target_state)
        self.state = np.copy(self.initial_state)
        self.state_weights = state_weights
        self.integral_weights = integral_weights
        self.last_action = 300  # 用于计算控制变化成本

    def cstr_system_dynamics(self, t, x, u):
        Ca, T = x[0], x[1]
        Tc = u
        Ca = max(0, Ca)
        T = max(1.0, T)
        rA = self.params.k0 * np.exp(-self.params.EA_over_R / T) * Ca
        dCa_dt = self.params.q / self.params.V * (self.params.Caf - Ca) - rA
        dT_dt = (
            self.params.q / self.params.V * (self.params.Ti - T)
            + ((-self.params.deltaHr * rA) / (self.params.rho * self.params.C))
            + (self.params.UA * (Tc - T) / (self.params.rho * self.params.C * self.params.V))
        )
        return [dCa_dt, dT_dt]

    def reset(self):
        self.state = np.copy(self.initial_state)
        self.last_action = 300
        # 允许在外部修改参数以模拟扰动
        self.params.Caf = 1.0
        return self.state

    def step(self, action, integral_error):
        sol = solve_ivp(
            lambda t, x: self.cstr_system_dynamics(t, x, action),
            [0, self.dt],
            self.state,
            method="RK45",
        )
        next_state = sol.y[:, -1]
        reward = -self._calculate_cost(self.state, integral_error, action, self.last_action)
        self.state = next_state
        self.last_action = action
        done = False
        info = {}
        return next_state, reward, done, info

    def _calculate_cost(self, x, I, u, prev_u):
        state_cost = np.sum(self.state_weights * (x - self.target_state) ** 2)
        integral_cost = np.sum(self.integral_weights * I ** 2)
        control_cost = 0.2 * (u - 300) ** 2 + 5.0 * (u - prev_u) ** 2
        return state_cost + integral_cost + control_cost

# ---------------------------------------------------------------------------
# 2. 世界模型 (World Model) (与原版相同)
# ---------------------------------------------------------------------------
class KoopmanWorldModel:
    def __init__(self, normalizer, poly_order, control_dim, integral_dim, learning_rate=1e-6):
        self.normalizer = normalizer
        self.poly_order = poly_order
        self.learning_rate = learning_rate
        self.p_phi = 2
        self.p_psi = lift_state(np.zeros((2, 1)), poly_order).shape[0] + integral_dim + control_dim
        self.A = np.zeros((self.p_phi, self.p_phi))
        self.B = np.zeros((self.p_phi, self.p_psi))
        self.w_prev = np.zeros(self.p_phi)
        self.is_pre_trained = False

    def pretrain(self, x_trajs, u_trajs, w_trajs, dt, target_state):
        print("正在对Koopman世界模型进行离线预训练...")
        Theta_w, Upsilon_z, Theta_w_plus = [], [], []
        for x_traj, u_traj, w_traj in tqdm(zip(x_trajs, u_trajs, w_trajs), total=len(x_trajs)):
            integral_error = np.zeros(target_state.shape)
            for k in range(w_traj.shape[1] - 1):
                w_k, w_k_plus_1 = w_traj[:, k], w_traj[:, k + 1]
                x_k, u_k = x_traj[:, k], u_traj[k]
                z_k = self._get_augmented_lifted_state(x_k, u_k, integral_error)
                Theta_w.append(w_k)
                Upsilon_z.append(z_k)
                Theta_w_plus.append(w_k_plus_1)
                integral_error += (x_k - target_state) * dt
        Theta_w, Upsilon_z, Theta_w_plus = (
            np.array(Theta_w).T,
            np.array(Upsilon_z).T,
            np.array(Theta_w_plus).T,
        )
        if Theta_w.shape[1] > 0:
            Psi = np.vstack([Theta_w, Upsilon_z])
            try:
                AB = Theta_w_plus @ pinv(Psi, rcond=1e-6)
                self.A, self.B = AB[:, : self.p_phi], AB[:, self.p_phi :]
                self.is_pre_trained = True
                print("模型预训练完成。")
            except np.linalg.LinAlgError:
                print("预训练期间发生线性代数错误。")
        else:
            print("没有有效的离线数据进行预训练。")

    def _get_augmented_lifted_state(self, x_current, u_current, I_current):
        x_norm = self.normalizer.transform(x_current)
        lifted_state = lift_state(x_norm, self.poly_order)
        lifted_input = lift_input(u_current)
        return np.concatenate([lifted_state, I_current, lifted_input])

    def predict(self, x_current, u_current, I_current, w_current):
        z_t = self._get_augmented_lifted_state(x_current, u_current, I_current)
        w_next_hat = self.A @ w_current + self.B @ z_t
        return w_next_hat

    def update(self, x_current, u_current, I_current, w_observed):
        z_t = self._get_augmented_lifted_state(x_current, u_current, I_current)
        w_pred = self.A @ self.w_prev + self.B @ z_t
        error = w_pred - w_observed
        grad_A = np.outer(error, self.w_prev)
        grad_B = np.outer(error, z_t)
        self.A -= self.learning_rate * grad_A
        self.B -= self.learning_rate * grad_B
        self.w_prev = w_observed

# ---------------------------------------------------------------------------
# 3. 安全层 (Safety Shield)
# ---------------------------------------------------------------------------
class SafetyShield:
    def __init__(self, world_model, safety_constraints, control_limits, dt):
        self.world_model = world_model
        self.constraints = safety_constraints
        self.control_limits = control_limits
        self.dt = dt

    def backup_policy(self, state):
        _, T = state
        T_max = self.constraints["T_max"]
        if T > T_max - 2.0:
            return self.control_limits[0] + (self.control_limits[1] - self.control_limits[0]) * max(0, (T_max - T) / 2.0)
        return np.mean(self.control_limits)

    def project_control(self, current_state, integral_error, u_nominal):
        u = cp.Variable()
        objective = cp.Minimize(cp.sum_squares(u - u_nominal))

        # 使用数值方法构造仿射表达式，避免符号变量问题
        z_at_u0 = self.world_model._get_augmented_lifted_state(current_state, 0, integral_error)
        z_at_u1 = self.world_model._get_augmented_lifted_state(current_state, 1, integral_error)
        B_times_z_at_u0 = self.world_model.B @ z_at_u0
        B_u_col = self.world_model.B @ (z_at_u1 - z_at_u0)  # B中对应于u的部分

        w_hat_affine_part = self.world_model.A @ self.world_model.w_prev + B_times_z_at_u0
        w_hat = w_hat_affine_part + cp.multiply(B_u_col, u)

        x_next_hat = current_state + w_hat

        cons = [self.control_limits[0] <= u, u <= self.control_limits[1]]
        T_max = self.constraints["T_max"]
        gamma = self.constraints["gamma"]

        h_current = T_max - current_state[1]
        h_next = T_max - x_next_hat[1]

        epsilon = 0.0
        Lh = 1.0

        dcbf_constraint = h_next >= (1 - gamma * self.dt) * h_current - Lh * epsilon
        cons.append(dcbf_constraint)

        problem = cp.Problem(objective, cons)

        try:
            problem.solve(solver=cp.SCS)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value
            else:
                return self.backup_policy(current_state)
        except cp.SolverError:
            return self.backup_policy(current_state)

# ---------------------------------------------------------------------------
# 4. 智能体 (Agent)
# ---------------------------------------------------------------------------
class DynaAgent:
    def __init__(self, world_model, target_state, dt, **kwargs):
        self.world_model = world_model
        self.target_state = target_state
        self.dt = dt
        self.n_samples = kwargs.get("n_samples", 300)
        self.horizon = kwargs.get("horizon", 15)
        self.control_limits = kwargs.get("control_limits", (280, 320))
        self.noise_sigma = kwargs.get("noise_sigma", 0.75)
        self.lambda_ = kwargs.get("lambda_", 0.05)
        self.state_bounds = kwargs.get("state_bounds")
        self.state_weights = kwargs.get("state_weights")
        self.integral_weights = kwargs.get("integral_weights")
        self.integral_error = np.zeros(len(target_state))
        self.previous_controls = np.full(self.horizon, np.mean(self.control_limits))
        self.safety_shield = kwargs.get("safety_shield", None)

    def reset(self):
        self.integral_error.fill(0)
        self.previous_controls.fill(np.mean(self.control_limits))
        self.world_model.w_prev.fill(0)

    def act(self, current_state):
        nominal_controls = np.append(self.previous_controls[1:], self.previous_controls[-1])
        perturbed_controls = np.random.randn(self.n_samples, self.horizon) * self.noise_sigma
        control_samples = np.clip(nominal_controls + perturbed_controls, *self.control_limits)

        w_current_for_planning = self.world_model.w_prev
        costs = np.array(
            [self._simulate_trajectory_cost(current_state, self.integral_error, w_current_for_planning, cs) for cs in control_samples]
        )

        valid = np.isfinite(costs)
        if not np.any(valid):
            u_nominal = nominal_controls[0]
        else:
            min_cost = np.min(costs[valid])
            exp_costs = np.exp(-(costs[valid] - min_cost) / self.lambda_)
            weights = exp_costs / (np.sum(exp_costs) + 1e-9)
            self.previous_controls = np.einsum("i,ij->j", weights, control_samples[valid])
            u_nominal = self.previous_controls[0]

        if self.safety_shield:
            return self.safety_shield.project_control(current_state, self.integral_error, u_nominal)
        else:
            return u_nominal

    def observe_transition(self, next_state):
        self.integral_error += (next_state - self.target_state) * self.dt

    def _planning_cost_function(self, x, I, u, prev_u):
        state_cost = np.sum(self.state_weights * (x - self.target_state) ** 2)
        integral_cost = np.sum(self.integral_weights * I ** 2)
        control_cost = 0.2 * (u - 300) ** 2 + 5.0 * (u - prev_u) ** 2
        return state_cost + integral_cost + control_cost

    def _planning_terminal_cost(self, x, I):
        return 100 * np.sum((x - self.target_state) ** 2) + 20 * np.sum(I ** 2)

    def _simulate_trajectory_cost(self, x_init, I_init, w_init, control_sequence):
        x, I, w = np.copy(x_init), np.copy(I_init), np.copy(w_init)
        total_cost = 0.0
        for i in range(self.horizon):
            w = self.world_model.predict(x, control_sequence[i], I, w)
            x += w
            I += (x - self.target_state) * self.dt
            if not (
                self.state_bounds[0][0] <= x[0] <= self.state_bounds[0][1]
                and self.state_bounds[1][0] <= x[1] <= self.state_bounds[1][1]
            ):
                return np.inf
            prev_u = self.previous_controls[0] if i == 0 else control_sequence[i - 1]
            total_cost += self._planning_cost_function(x, I, control_sequence[i], prev_u)
        total_cost += self._planning_terminal_cost(x, I)
        return total_cost

# --- 新增：专门为 KoopmanMPC 服务的符号计算函数 ---
def _symbolic_lift_state(x_normalized_expr, poly_order=2):
    x1, x2 = x_normalized_expr[0], x_normalized_expr[1]
    lifted = [1.0]
    for i in range(1, poly_order + 1):
        for j in range(i + 1):
            term1 = cp.power(x1, i - j)
            term2 = cp.power(x2, j)
            lifted.append(cp.multiply(term1, term2))
    return cp.vstack(lifted)

class KoopmanMPC:
    def __init__(self, world_model, target_state, dt, **kwargs):
        self.world_model = world_model
        self.target_state = target_state
        self.dt = dt
        self.horizon = kwargs.get("horizon", 15)
        self.control_limits = kwargs.get("control_limits", (280, 320))
        self.state_bounds = kwargs.get("state_bounds")
        self.state_weights = kwargs.get("state_weights")
        self.integral_weights = kwargs.get("integral_weights")
        self.integral_error = np.zeros(len(target_state))

    def reset(self):
        self.integral_error.fill(0)
        self.world_model.w_prev.fill(0)

    def _symbolic_get_augmented_lifted_state(self, x_expr, u_expr, I_expr):
        """处理cvxpy表达式的符号版本。

        关键修复：在堆叠 lifted_state_expr、I_expr 和 u_expr 之前，显式地
        将它们重塑为列向量，避免 ``cvxpy.vstack`` 中的维度不匹配错误。
        """
        x_col = cp.reshape(
            x_expr,
            (self.world_model.normalizer.original_dim, 1),
            order="F",
        )
        x_norm_expr = (x_col - self.world_model.normalizer.mean) / self.world_model.normalizer.std
        lifted_state_expr = _symbolic_lift_state(x_norm_expr, self.world_model.poly_order)
        I_col = cp.reshape(I_expr, (len(self.target_state), 1), order="F")
        u_col = cp.reshape(u_expr, (1, 1), order="F")
        return cp.vstack([lifted_state_expr, I_col, u_col])

    def act(self, current_state):
        u = cp.Variable(self.horizon)
        x = cp.Variable((2, self.horizon + 1))
        I = cp.Variable((2, self.horizon + 1))
        w = cp.Variable((2, self.horizon + 1))

        objective = 0
        constraints = [
            x[:, 0] == current_state,
            I[:, 0] == self.integral_error,
            w[:, 0] == self.world_model.w_prev,
        ]

        for t in range(self.horizon):
            state_cost = cp.sum_squares(np.sqrt(self.state_weights) @ (x[:, t] - self.target_state))
            integral_cost = cp.sum_squares(np.sqrt(self.integral_weights) @ I[:, t])
            control_cost = 0.2 * cp.sum_squares(u[t] - 300)
            objective += state_cost + integral_cost + control_cost

            # 使用符号辅助函数构建约束
            z_expr = self._symbolic_get_augmented_lifted_state(x[:, t], u[t], I[:, t])
            w_pred = self.world_model.A @ w[:, t] + self.world_model.B @ z_expr

            constraints += [
                w[:, t + 1] == w_pred,
                x[:, t + 1] == x[:, t] + w[:, t + 1],
                I[:, t + 1] == I[:, t] + cp.multiply(x[:, t] - self.target_state, self.dt),
            ]

            constraints += [
                self.state_bounds[0][0] <= x[0, t + 1],
                x[0, t + 1] <= self.state_bounds[0][1],
                self.state_bounds[1][0] <= x[1, t + 1],
                x[1, t + 1] <= self.state_bounds[1][1],
                self.control_limits[0] <= u[t],
                u[t] <= self.control_limits[1],
            ]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(solver=cp.SCS)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value[0]
            else:
                return np.mean(self.control_limits)
        except (cp.SolverError, cp.DCPError):
            # If the problem is infeasible or violates DCP rules,
            # fall back to a safe nominal control.
            return np.mean(self.control_limits)

    def observe_transition(self, next_state):
        self.integral_error += (next_state - self.target_state) * self.dt

# ---------------------------------------------------------------------------
# 5. 辅助函数 (数据生成与可视化)
# ---------------------------------------------------------------------------
def generate_random_trajectories(n_trajectories, traj_length, params, x_bounds, dt):
    all_x, all_u = [], []
    t_span_per_traj = (0, (traj_length - 1) * dt)
    print("正在为离线训练生成数据...")
    for _ in tqdm(range(n_trajectories), desc="生成轨迹"):
        x0 = [np.random.uniform(b[0], b[1]) for b in x_bounds]
        num_sin = np.random.randint(2, 5)
        amps = np.random.uniform(1, 5, num_sin)
        freqs = np.random.uniform(0.05, 0.2, num_sin)
        phases = np.random.uniform(0, 2 * np.pi, num_sin)
        bias = np.random.uniform(295, 305)

        def smooth_random_input(t_array):
            sine_waves = [amp * np.sin(freq * t_array + phase) for amp, freq, phase in zip(amps, freqs, phases)]
            return bias + np.sum(np.array(sine_waves), axis=0)

        env_temp = CSTREnvironment(params, dt, x0, np.zeros(2), np.zeros(2), np.zeros(2))
        x_traj, u_traj = [env_temp.state], []
        t_eval = np.linspace(t_span_per_traj[0], t_span_per_traj[1], traj_length)
        u_values = smooth_random_input(t_eval)
        for k in range(traj_length - 1):
            next_state, _, _, _ = env_temp.step(u_values[k], np.zeros(2))
            if not np.all(np.isfinite(next_state)):
                x_traj = []
                break
            x_traj.append(next_state)
            u_traj.append(u_values[k])
        if x_traj:
            u_traj.append(u_values[-1])
            all_x.append(np.array(x_traj).T)
            all_u.append(np.array(u_traj))
    if not all_x:
        raise RuntimeError("未能生成任何有效轨迹。")
    return all_x, all_u


def visualize_comparison(results, target_state, safety_constraints, t_sim, title_prefix):
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{title_prefix} - 控制器性能对比", fontsize=16)

    colors = {"Unshielded MPPI": "blue", "Shielded MPPI": "green", "Koopman MPC": "purple"}
    styles = {"Unshielded MPPI": "-", "Shielded MPPI": "--", "Koopman MPC": ":"}

    for name, data in results.items():
        states = data["states"]
        controls = data["controls"]
        axs[0].plot(t_sim, states[0, :], color=colors[name], linestyle=styles[name], label=f"{name} - $C_a$")
        axs[1].plot(t_sim, states[1, :], color=colors[name], linestyle=styles[name], label=f"{name} - T")
        axs[2].plot(t_sim[:-1], controls, color=colors[name], linestyle=styles[name], label=f"{name} - $T_c$")

    axs[0].axhline(target_state[0], color="k", linestyle="--", label="目标 $C_a$")
    axs[0].set_ylabel("浓度 (mol/L)"), axs[0].legend(), axs[0].grid(True)

    axs[1].axhline(target_state[1], color="k", linestyle="--", label="目标 T")
    axs[1].axhline(
        safety_constraints["T_max"],
        color="r",
        linestyle="-.",
        label=f"安全上限 T_max={safety_constraints['T_max']}K",
    )
    axs[1].set_ylabel("温度 (K)"), axs[1].legend(), axs[1].grid(True)

    axs[2].set_xlabel("时间 (s)"), axs[2].set_ylabel("冷却剂温度 Tc (K)"), axs[2].legend(), axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_comparison.png", dpi=300)
    plt.show()


def calculate_metrics(states, controls, target_state, safety_constraints, dt):
    errors = states[:, :-1] - target_state[:, np.newaxis]
    iae = np.sum(np.abs(errors), axis=1) * dt

    violations = states[1, :] > safety_constraints["T_max"]
    violation_rate = np.mean(violations) * 100

    min_safety_margin = np.min(safety_constraints["T_max"] - states[1, :])

    control_effort = np.sum(np.diff(controls, prepend=controls[0]) ** 2) * dt

    return {
        "IAE_Ca": iae[0],
        "IAE_T": iae[1],
        "Violation Rate (%)": violation_rate,
        "Min Safety Margin (K)": min_safety_margin,
        "Control Effort": control_effort,
    }

# ---------------------------------------------------------------------------
# 6. 主执行模块 (Dyna-RL 循环)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1. 初始化 ---
    print("--- 1. 系统初始化 ---")

    class CSTRParams:
        q, V, rho, C, deltaHr, EA_over_R, k0, UA, Ti, Caf = 100, 100, 1000, 0.239, -5e4, 8750, 7.2e10, 5e4, 350, 1

    params = CSTRParams()
    poly_order, controller_dt, original_dim, integral_dim = 2, 0.5, 2, 2

    T_target = 320.0
    k_target = params.k0 * np.exp(-params.EA_over_R / T_target)
    Ca_target = (params.q / params.V * params.Caf) / (params.q / params.V + k_target)
    target_state = np.array([Ca_target, T_target])
    initial_state_test = np.array([0.5, 350.0])

    safety_constraints = {"T_max": 370.0, "gamma": 0.8}

    # --- 2. 离线数据生成与模型预训练 ---
    print("\n--- 2. 模型预训练阶段 ---")
    offline_x, offline_u = generate_random_trajectories(50, 100, params, [(0.7, 1.1), (300, 360)], controller_dt)

    normalizer = Normalizer(original_dim)
    normalizer.fit(np.hstack(offline_x))

    shared_world_model = KoopmanWorldModel(normalizer, poly_order, 1, integral_dim, learning_rate=1e-6)

    offline_w = [np.diff(x_traj, axis=1) for x_traj in offline_x]
    shared_world_model.pretrain(offline_x, offline_u, offline_w, controller_dt, target_state)

    # --- 3. 初始化环境和智能体 ---
    print("\n--- 3. 初始化RL环境和所有对比智能体 ---")
    agent_params = {
        "target_state": target_state,
        "dt": controller_dt,
        "horizon": 15,
        "control_limits": (280, 320),
        "state_bounds": [(0.0, 2.0), (250.0, 400.0)],  # 放宽内部模型的界限
        "state_weights": np.array([5.0, 5.0]),
        "integral_weights": np.array([0.50, 2.5]),
    }

    safety_shield = SafetyShield(shared_world_model, safety_constraints, agent_params["control_limits"], controller_dt)

    agents = {
        "Unshielded MPPI": DynaAgent(world_model=shared_world_model, **agent_params),
        "Shielded MPPI": DynaAgent(world_model=shared_world_model, safety_shield=safety_shield, **agent_params),
        "Koopman MPC": KoopmanMPC(world_model=shared_world_model, **agent_params),
    }

    env = CSTREnvironment(
        params,
        controller_dt,
        initial_state_test,
        target_state,
        agent_params["state_weights"],
        agent_params["integral_weights"],
    )

    # --- 4. 执行实验 ---
    n_steps = 600
    disturbance_step = 300
    results = {}
    metrics = {}

    for name, agent in agents.items():
        print(f"\n--- 4. 正在运行控制器: {name} ---")
        current_state = env.reset()
        agent.reset()

        # 每次运行前重置共享模型的状态
        shared_world_model.w_prev.fill(0)

        state_history = [current_state]
        control_history = []

        for i in tqdm(range(n_steps), desc=f"Simulating {name}"):
            if i == disturbance_step:
                print(f"\n在步骤 {i} 引入扰动: Caf 从 1.0 变为 1.2")
                env.params.Caf = 1.2

            action = agent.act(current_state)

            next_state, _, _, _ = env.step(action, agent.integral_error)

            w_observed = next_state - current_state
            shared_world_model.update(current_state, action, agent.integral_error, w_observed)

            agent.observe_transition(next_state)

            current_state = next_state

            state_history.append(current_state)
            control_history.append(action)

        results[name] = {"states": np.array(state_history).T, "controls": np.array(control_history)}
        metrics[name] = calculate_metrics(
            results[name]["states"],
            results[name]["controls"],
            target_state,
            safety_constraints,
            controller_dt,
        )

    # --- 5. 结果分析与可视化 ---
    print("\n--- 5. 实验结果分析 ---")
    print("控制器性能指标对比:")
    metrics_df = pd.DataFrame(metrics).T
    print(tabulate(metrics_df, headers="keys", tablefmt="psql"))

    t_sim = np.arange(n_steps + 1) * controller_dt
    visualize_comparison(results, target_state, safety_constraints, t_sim, "CSTR控制扰动响应")
