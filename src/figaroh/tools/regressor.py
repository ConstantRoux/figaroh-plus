"""Regressor matrix computation utilities for robot dynamic identification."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from scipy.optimize import lsq_linear
import cvxpy as cp

@dataclass
class RegressorConfig:
    """Configuration for regressor computation."""
    has_friction: bool = False
    has_actuator_inertia: bool = False
    has_joint_offset: bool = False
    is_joint_torques: bool = True
    is_external_wrench: bool = False
    force_torque: Optional[List[str]] = None
    additional_columns: int = 0


class RegressorBuilder:
    """Enhanced regressor builder with better organization."""

    def __init__(self, robot, config: Optional[RegressorConfig] = None):
        self.robot = robot
        self.config = config or RegressorConfig()
        self.nv = robot.model.nv
        self.nonzero_inertias = self._get_nonzero_inertias()

    def build_basic_regressor(self, q: np.ndarray, v: np.ndarray, a: np.ndarray, identif_config=None) -> np.ndarray:
        """Build basic regressor matrix."""
        # Normalize inputs
        Q, V, A, N = self._normalize_inputs(q, v, a)

        if self.config.is_joint_torques:
            return self._build_joint_torque_regressor(Q, V, A, N, identif_config)
        elif self.config.is_external_wrench:
            return self._build_external_wrench_regressor(Q, V, A, N, identif_config)
        else:
            raise ValueError("Must specify either joint_torques or external_wrench mode")

    def _normalize_inputs(self, q, v, a) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Normalize and validate inputs."""
        Q = self._ensure_2d(q, self.robot.model.nq, "q")
        V = self._ensure_2d(v, self.nv, "v") 
        A = self._ensure_2d(a, self.nv, "a")

        N = Q.shape[0]
        if V.shape[0] != N or A.shape[0] != N:
            raise ValueError(f"Inconsistent sample counts: q={N}, v={V.shape[0]}, a={A.shape[0]}")

        return Q, V, A, N

    def _ensure_2d(self, x, expected_width: int, name: str) -> np.ndarray:
        """Ensure input is 2D array with correct width."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != expected_width:
            raise ValueError(f"{name} must have {expected_width} columns, got {x.shape[1]}")
        return x

    def _get_nonzero_inertias(self) -> List[int]:
        """Get indices of bodies with non-zero mass."""
        return [i for i, inertia in enumerate(self.robot.model.inertias.tolist()) 
                if inertia.mass != 0]

    def _build_joint_torque_regressor(self, Q, V, A, N, identif_config=None) -> np.ndarray:
        """Build regressor for joint torque identification."""
        W_ = np.zeros([N * self.nv, (10 + self.config.additional_columns) * self.nv])

        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                self.robot.model, self.robot.data, Q[i], V[i], A[i]
            )
            self._fill_joint_regressor_sample(W_, W_temp, V, A, i, N, identif_config)
        return W_
        # return self._reorder_parameters(W_, self.nv)

    def _build_external_wrench_regressor(self, Q, V, A, N, identif_config=None) -> np.ndarray:
        """Build regressor for external wrench identification."""
        nb_bodies = len(self.robot.model.inertias) - 1
        ft_components = self.config.force_torque or []

        W_ = np.zeros([N * 6, (10 + self.config.additional_columns) * nb_bodies])

        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                self.robot.model, self.robot.data, Q[i], V[i], A[i]
            )
            self._fill_wrench_regressor_sample(W_, W_temp, V, A, ft_components, i, N, nb_bodies, identif_config)
        return W_
        # return self._reorder_parameters(W, nb_bodies)

    def _fill_joint_regressor_sample(self, W, W_temp, V, A, sample_idx, N, identif_config=None):
        n_add = 0
        if self.config.has_friction: n_add += 2
        if self.config.has_actuator_inertia: n_add += 1
        if self.config.has_joint_offset: n_add += 1
        
        stride = 10 + n_add
        
        for j in range(W_temp.shape[0]):
            base_idx = j * N + sample_idx
            joint_id = j + 1
            for k in range(self.nv):
                src_start = k * 10
                src_end   = (k + 1) * 10
                dst_start = k * stride
                dst_end   = dst_start + 10
                W[base_idx, dst_start:dst_end] = W_temp[j, src_start:src_end]
                
                if k == j:
                    current_col = dst_end
                    
                    # 1. Friction (fv, fs)
                    if self.config.has_friction:
                        if joint_id in identif_config["act_idxv"]:
                            W[base_idx, current_col]     = V[sample_idx, j]           # fv
                            W[base_idx, current_col + 1] = np.sign(V[sample_idx, j])  # fs
                        current_col += 2

                    # 2. Actuator Inertia (ia)
                    if self.config.has_actuator_inertia:
                        if j in identif_config["act_idxv"]:
                            W[base_idx, current_col] = A[sample_idx, j]               # ia
                        current_col += 1
                        
                    # 3. Joint Offset (off)
                    if self.config.has_joint_offset:
                        if j in identif_config["act_idxv"]:
                            W[base_idx, current_col] = 1.0                            # offset
                        current_col += 1

    def _fill_wrench_regressor_sample(self, W, W_temp, V, A, ft_components, sample_idx, N, nb_bodies, identif_config=None):
        """Fill regressor for one sample in external wrench mode."""
        for k, ft in enumerate(ft_components):
            j = "Fx Fy Fz Mx My Mz".split().index(ft)
            base_idx = j * N + sample_idx
            W[base_idx, :10 * nb_bodies] = W_temp[j, :10 * nb_bodies]

        for j in range(nb_bodies):
            base_idx = j * 6 + sample_idx

            if identif_config and j in identif_config["act_idxv"]:
                if self.config.has_friction:
                    W[base_idx, 10 * nb_bodies + j] = V[sample_idx, j]  # fv
                    W[base_idx, 10 * nb_bodies + nb_bodies + j] = np.sign(V[sample_idx, j])  # fs

                if self.config.has_actuator_inertia:
                    W[base_idx, 10 * nb_bodies + 2 * nb_bodies + j] = A[sample_idx, j]  # ia

                if self.config.has_joint_offset:
                    W[base_idx, 10 * nb_bodies + 2 * nb_bodies + nb_bodies + j] = 1  # offset

    def _reorder_parameters(self, W: np.ndarray, num_params: int) -> np.ndarray:
        """Reorder parameters to standard format."""
        cols = 10 + self.config.additional_columns
        W_reordered = np.zeros([W.shape[0], cols * num_params])

        # Parameter order: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m, ia, fv, fs, offset]
        param_order = [4, 5, 7, 6, 8, 9, 1, 2, 3, 0]  # Pinocchio to standard order

        for k in range(num_params):
            base_out = k * cols
            base_in = k * 10

            # Reorder inertial parameters
            for i, old_idx in enumerate(param_order):
                W_reordered[:, base_out + i] = W[:, base_in + old_idx]

            # Add additional parameters
            if self.config.additional_columns > 0:
                param_start = 10 * num_params
                W_reordered[:, base_out + 10] = W[:, param_start + 2*num_params + k]  # ia
                W_reordered[:, base_out + 11] = W[:, param_start + 2*k]  # fv
                W_reordered[:, base_out + 12] = W[:, param_start + 2*k + 1]  # fs
                W_reordered[:, base_out + 13] = W[:, param_start + 2*num_params + num_params + k]  # offset

        return W_reordered


# Backward compatibility functions
def build_regressor_basic(robot, q, v, a, identif_config, tau=None):
    """Legacy function for backward compatibility."""
    # Calculate additional columns based on enabled options
    additional_columns = sum([
        2 if identif_config.get("has_friction", False) else 0,  # fv and fs
        1 if identif_config.get("has_actuator_inertia", False) else 0,  # ia
        1 if identif_config.get("has_joint_offset", False) else 0,  # offset
    ])
    
    config = RegressorConfig(
        has_friction=identif_config.get("has_friction", False),
        has_actuator_inertia=identif_config.get("has_actuator_inertia", False),
        has_joint_offset=identif_config.get("has_joint_offset", False),
        is_joint_torques=identif_config.get("is_joint_torques", True),
        is_external_wrench=identif_config.get("is_external_wrench", False),
        force_torque=identif_config.get("force_torque", None),
        additional_columns=additional_columns
    )
    
    builder = RegressorBuilder(robot, config)
    return builder.build_basic_regressor(q, v, a, identif_config)


# Keep other functions with minor improvements...
def eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
    """Eliminate columns with small L2 norm."""
    col_norms = np.diag(W.T @ W)
    param_keys = list(params_std.keys())
    
    keep_indices = []
    keep_params = []
    
    for i, norm in enumerate(col_norms):
        if norm >= tol_e and i < len(param_keys):
            keep_indices.append(i)
            keep_params.append(param_keys[i])
    
    W_reduced = W[:, keep_indices]
    return W_reduced, keep_params


def get_index_eliminate(W, params_std, tol_e=1e-6):
    """Get indices of columns to eliminate based on tolerance.

    Args:
        W: Joint torque regressor matrix
        params_std: Standard parameters dictionary
        tol_e: Tolerance value

    Returns:
        tuple:
            - List of indices to eliminate
            - List of remaining parameters
    """
    col_norm = np.diag(np.dot(W.T, W))
    idx_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
        else:
            params_r.append(list(params_std.keys())[i])
    return idx_e, params_r


def build_regressor_reduced(W, idx_e):
    """Build reduced regressor matrix.

    Args:
        W: Input regressor matrix
        idx_e: Indices of columns to eliminate

    Returns:
        ndarray: Reduced regressor matrix
    """
    W_e = np.delete(W, idx_e, 1)
    return W_e



def build_total_regressor_current_tls(
    W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, param
):
    """
    Standard Total Least Squares (TLS) implementation.
    Identifies gains and inertial parameters simultaneously.
    """
    n_rows_u = len(I_u)
    n_rows_l = len(I_l) 
    nb_j = len(param['act_idxv'])
    n_samples_u = n_rows_u // nb_j
    n_samples_l = n_rows_l // nb_j

    # --- 1. Base Regressor (Kinematics) ---
    # Stack: [ -W_base_Unloaded ]
    #        [ -W_base_Loaded   ]
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)
    
    # --- 2. Currents (Gains) ---
    # Build Block Diagonal Matrices
    V_a_list = []
    for ii in range(nb_j):
        col_vec = np.zeros((n_rows_u, 1))
        start = ii * n_samples_u
        end   = (ii + 1) * n_samples_u
        col_vec[start:end] = I_u[start:end].reshape(n_samples_u, 1)
        V_a_list.append(col_vec)
    V_a = np.hstack(V_a_list)

    V_b_list = []
    for ii in range(nb_j):
        col_vec = np.zeros((n_rows_l, 1))
        start = ii * n_samples_l
        end   = (ii + 1) * n_samples_l
        col_vec[start:end] = I_l[start:end].reshape(n_samples_l, 1)
        V_b_list.append(col_vec)
    V_b = np.hstack(V_b_list)

    # Stack Currents: [ V_a ]
    #                 [ V_b ]
    W_current = np.concatenate((V_a, V_b), axis=0)
    W_tot = np.concatenate((W_tot, W_current), axis=1)
    
    # --- 3. Payload Regressor (Kinematics) ---
    cols_per_joint = 10
    if param.get('has_friction', False): cols_per_joint += 2
    if param.get('has_actuator_inertia', False): cols_per_joint += 1
    if param.get('has_joint_offset', False): cols_per_joint += 1
    
    idx_body = param['which_body_loaded'] - 1 
    start_col = idx_body * cols_per_joint
    cols_to_keep = [1,2,3,4,5,6,7,8,9]
    
    W_l_temp = np.zeros((len(W_l), len(cols_to_keep)))
    for i, k in enumerate(cols_to_keep):
        W_l_temp[:, i] = W_l[:, start_col + k]

    W_e_l = W_l_temp
        
    # Stack Unknown Payload: [ 0 ]
    #                        [ -W_e_l ]
    zeros_padding = np.zeros((n_rows_u, W_e_l.shape[1])) 
    W_upayload = np.concatenate((zeros_padding, -W_e_l), axis=0)
    W_tot = np.concatenate((W_tot, W_upayload), axis=1)
    
    # --- 4. Known Mass (Reference) ---
    mass_col_idx = 0 
    col_mass = -W_l[:, start_col + mass_col_idx].reshape(len(W_l), 1)
    
    zeros_mass = np.zeros((n_rows_u, 1))
    W_kpayload = np.concatenate((zeros_mass, col_mass), axis=0)
    W_tot = np.concatenate((W_tot, W_kpayload), axis=1)
    
    # --- 5. Solve via SVD ---
    print("[WTLS] Matrix Rank:", np.linalg.matrix_rank(W_tot), "Shape:", W_tot.shape)
    
    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = Vh[-1, :]
    
    # Normalize with respect to the known mass (last element)
    V_norm = param['mass_load'] * np.divide(V[:], V[-1])     
    residue = np.matmul(W_tot, V_norm)
            
    return W_tot, V_norm, residue

def build_total_regressor_wrench(
    W_b_u, W_b_l, W_l, tau_u, tau_l, param_standard_l, param
):
    """Build regressor for total least squares with external wrench measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        tau_u: External wrench in unloaded case
        tau_l: External wrench in loaded case
        param_standard_l: Standard parameters in loaded case
        param: Dictionary of settings

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized parameter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    tau_meast_ul = np.reshape(tau_u, (len(tau_u), 1))
    tau_meast_l = np.reshape(tau_l, (len(tau_l), 1))

    nb_samples_ul = int(len(tau_meast_ul) / 6)
    nb_samples_l = int(len(tau_meast_l) / 6)

    tau_ul = np.concatenate([
        tau_meast_ul[:nb_samples_ul],
        np.zeros((len(tau_meast_ul) - nb_samples_ul, 1))
    ], axis=0)
    
    tau_l = np.concatenate([
        tau_meast_l[:nb_samples_l],
        np.zeros((len(tau_meast_l) - nb_samples_l, 1))
    ], axis=0)

    for ii in range(1, 6):
        tau_ul_ii = np.concatenate([
            np.concatenate([
                np.zeros((nb_samples_ul * ii, 1)),
                tau_meast_ul[
                    nb_samples_ul * ii:(ii + 1) * nb_samples_ul
                ]
            ], axis=0),
            np.zeros((nb_samples_ul * (5 - ii), 1))
        ], axis=0)

        tau_l_ii = np.concatenate([
            np.concatenate([
                np.zeros((nb_samples_l * ii, 1)),
                tau_meast_l[
                    nb_samples_l * ii:(ii + 1) * nb_samples_l
                ]
            ], axis=0),
            np.zeros((nb_samples_l * (5 - ii), 1))
        ], axis=0)

        tau_ul = np.concatenate((tau_ul, tau_ul_ii), axis=1)
        tau_l = np.concatenate((tau_l, tau_l_ii), axis=1)

    W_tau = np.concatenate((tau_ul, tau_l), axis=0)
    W_tot = np.concatenate((W_tot, W_tau), axis=1)

    W_l_temp = np.zeros((len(W_l), 9))
    for k in range(9):
        W_l_temp[:, k] = W_l[
            :, (identif_config["which_body_loaded"]) * 10 + k
        ]
    W_upayload = np.concatenate(
        (np.zeros((len(W_l), W_l_temp.shape[1])), -W_l_temp),
        axis=0
    )
    W_tot = np.concatenate((W_tot, W_upayload), axis=1)
    
    W_kpayload = np.concatenate([
        np.zeros((len(W_l), 1)),
        -W_l[:, identif_config["which_body_loaded"] * 10 + 9].reshape(len(W_l), 1)
    ], axis=0)
    W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = identif_config["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue

def build_total_regressor_current_ols(
    W_b_u, W_b_l, W_l, I_u, I_l, param, bounds
):
    """
    Robust Constrained OLS implementation.
    Includes NaN checks and uses 'bvls' solver for stability.
    """
    # --- 1. Dimensions & Pre-checks ---
    if param.get('mass_load') is None:
        raise ValueError("param['mass_load'] is None! You must define the payload mass.")

    nb_j = len(param['act_idxv'])
    n_rows_u = len(I_u)
    n_rows_l = len(I_l) 
    n_samples_u = n_rows_u // nb_j
    n_samples_l = n_rows_l // nb_j

    # --- 2. Construct Regressor Components ---
    
    # A. Base Regressors (Kinematics)
    W_base_stack = np.concatenate((-W_b_u, -W_b_l), axis=0)
    
    # B. Currents (Gains)
    V_a_list = []
    for ii in range(nb_j):
        col_vec = np.zeros((n_rows_u, 1))
        start = ii * n_samples_u
        end   = (ii + 1) * n_samples_u
        col_vec[start:end] = I_u[start:end].reshape(n_samples_u, 1)
        V_a_list.append(col_vec)
    V_a = np.hstack(V_a_list)

    V_b_list = []
    for ii in range(nb_j):
        col_vec = np.zeros((n_rows_l, 1))
        start = ii * n_samples_l
        end   = (ii + 1) * n_samples_l
        col_vec[start:end] = I_l[start:end].reshape(n_samples_l, 1)
        V_b_list.append(col_vec)
    V_b = np.hstack(V_b_list)
    W_curr_stack = np.concatenate((V_a, V_b), axis=0)
    
    # C. Payload Unknowns
    cols_per_joint = 10
    if param.get('has_friction', False): cols_per_joint += 2
    if param.get('has_actuator_inertia', False): cols_per_joint += 1
    if param.get('has_joint_offset', False): cols_per_joint += 1
    
    idx_body = param['which_body_loaded'] - 1 
    start_col = idx_body * cols_per_joint
    cols_to_keep = [1,2,3,4,5,6,7,8,9]
    
    W_l_temp = np.zeros((len(W_l), len(cols_to_keep)))
    for i, k in enumerate(cols_to_keep):
        W_l_temp[:, i] = W_l[:, start_col + k]

    zeros_padding = np.zeros((n_rows_u, W_l_temp.shape[1])) 
    W_load_stack = np.concatenate((zeros_padding, -W_l_temp), axis=0)

    # D. Known Mass Column (The Forcing Term)
    mass_col_idx = 0 
    col_mass = -W_l[:, start_col + mass_col_idx].reshape(len(W_l), 1)
    zeros_mass = np.zeros((n_rows_u, 1))
    W_mass_stack = np.concatenate((zeros_mass, col_mass), axis=0)

    # --- 3. Build X and Y ---
    X = np.concatenate((W_base_stack, W_curr_stack, W_load_stack), axis=1)
    Y = (-W_mass_stack * param['mass_load']).flatten()
    
    # --- 4. DEBUG: NaN Checks ---
    if np.isnan(X).any():
        print(f"[ERROR] Regressor X contains {np.isnan(X).sum()} NaNs!")
        # Optional: Replace NaNs with 0 to allow solving (risky but functional)
        X = np.nan_to_num(X, nan=0.0)
    
    if np.isnan(Y).any():
        print(f"[ERROR] Target Y contains {np.isnan(Y).sum()} NaNs! Check W_l or mass_load.")
        Y = np.nan_to_num(Y, nan=0.0)

    # --- 5. Bounds Setup ---
    # Gains: Unbounded (-inf, inf)
    bounds_gains = [(0.0, np.inf)] * nb_j
    
    # Payload: 9 params [MX, MY, MZ, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    bounds_payload = [
        (-np.inf, np.inf), # 1: MX
        (-np.inf, np.inf), # 2: MY
        (-np.inf, np.inf), # 3: MZ
        (0.0,  np.inf),    # 4: Ixx (Positive)
        (-np.inf, np.inf), # 5: Ixy
        (-np.inf, np.inf), # 6: Ixz
        (0.0,  np.inf),    # 7: Iyy (Positive)
        (-np.inf, np.inf), # 8: Iyz
        (0.0,  np.inf),    # 9: Izz (Positive)
    ]
    
    full_bounds = bounds + bounds_gains + bounds_payload
    
    if len(full_bounds) != X.shape[1]:
         raise ValueError(f"Bounds mismatch: X has {X.shape[1]} cols, but {len(full_bounds)} bounds provided.")

    lower_bounds = np.array([b[0] for b in full_bounds])
    upper_bounds = np.array([b[1] for b in full_bounds])
    
    # --- 6. Solve (Using BVLS) ---
    print(f"[Constrained OLS] Solving for {X.shape[1]} variables...")
    
    # 'bvls' is generally more robust to scaling issues than 'trf'
    res = lsq_linear(X, Y, bounds=(lower_bounds, upper_bounds), method='bvls', verbose=0)
    
    if not res.success:
        print(f"[Warning] Optimization issue: {res.message}")
        
    Beta = res.x

    # --- 7. Output ---
    V_norm = np.append(Beta, param['mass_load'])
    W_tot = np.concatenate((X, W_mass_stack), axis=1)
    residue = np.matmul(W_tot, V_norm)
        
    return W_tot, V_norm, residue

def solve_LMI_OLS(
    W,  # The standard regressor from Pinocchio (N x 10*nb_joints)
    Y,               # Measured Torques (N x 1)
    nb_joints,
    phi_prior,
    lambda_reg
):
    total_params = 13 * nb_joints
    phi = cp.Variable(total_params)
    
    constraints = []
    
    for j in range(nb_joints):
        base = j * 13
        
        m   = phi[base + 0]
        h   = phi[base+1 : base+4]
        
        Ixx = phi[base + 4]
        Ixy = phi[base + 5]
        Iyy = phi[base + 6]
        Ixz = phi[base + 7]
        Iyz = phi[base + 8]
        Izz = phi[base + 9]
        
        I_3x3 = cp.bmat([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        
        tr_I = cp.trace(I_3x3)
        P_block = 0.5 * tr_I * np.eye(3) - I_3x3
        
        LMI_matrix = cp.bmat([
            [P_block,    cp.reshape(h, (3,1))],
            [cp.reshape(h, (1,3)), cp.reshape(m, (1,1))]
        ])
        
        constraints.append(LMI_matrix >> 0)
        
        Fv = phi[base + 10]
        Fs = phi[base + 11]
        
        constraints.append(Fv >= 0)
        constraints.append(Fs >= 0)
    
    N_samples = Y.shape[0]
    
    # Least Squares Error + Regularization
    cost = (1.0 / N_samples) * cp.sum_squares(Y - W @ phi) + \
       lambda_reg * cp.sum_squares(phi - phi_prior)
    
    # --- 5. Solve ---
    print(f"[SDP] Solving Full Robot ID with Friction ({total_params} vars)...")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    # SCS is a good splitting solver for large SDPs
    prob.solve(solver=cp.CLARABEL, verbose=True)
    
    if prob.status != 'optimal':
        print(f"[Warning] Solver status: {prob.status}")
        
    return phi.value

def solve_differential_LMI_OLS(
    W_A, Y_A_meas, mass_load_A, 
    W_B, Y_B_meas, mass_load_B,
    load_joint_idx,   
    nb_joints,
    joint_names,       
    phi_cad_dict,   
    armature_vals,
    mesh_bounds,
    lambda_r=0.1, 
    lambda_l=0.1,
):
    # --- 1. Data Prep ---
    Y_A_flat = Y_A_meas.flatten(order='F')
    Y_B_flat = Y_B_meas.flatten(order='F')
    nSamples_A = Y_A_meas.shape[0]
    nSamples_B = Y_B_meas.shape[0]
    
    # --- 2. Extract Payload Regressors ---
    start_col = (load_joint_idx-1) * 14 
    
    # Extract Mass Columns (Scaling factor for known mass)
    W_load_mass_A = W_A[:, start_col + 0]  
    W_load_mass_B = W_B[:, start_col + 0]
    
    # Extract Geometry Columns (Shape to identify: hx...Izz)
    # Columns 1 to 10 correspond to h(3) + I(6)
    W_load_geom_A = W_A[:, start_col + 1 : start_col + 10] 
    W_load_geom_B = W_B[:, start_col + 1 : start_col + 10] 
    
    # Pre-calculate known mass torque to move to RHS
    tau_payload_mass_A = W_load_mass_A * mass_load_A
    tau_payload_mass_B = W_load_mass_B * mass_load_B

    # --- 3. Build Scale Regressor (Diagonal Matrices) ---
    def build_scale_matrix(Y_flat, n_samples, n_joints):
        K = np.zeros((len(Y_flat), n_joints))
        for j in range(n_joints):
            start, end = j*n_samples, (j+1)*n_samples
            K[start:end, j] = Y_flat[start:end]
        return K

    K_A = build_scale_matrix(Y_A_flat, nSamples_A, nb_joints)
    K_B = build_scale_matrix(Y_B_flat, nSamples_B, nb_joints)

    # --- 4. Stack Linear System (Block Diagonal for Payloads) ---
    
    # System Structure:
    # [ -Robot_A | -Geom_A |   0    | Scale_A ] * [phi_r]
    # [ -Robot_B |   0    | -Geom_B | Scale_B ]   [phi_l_A]
    #                                            [phi_l_B]
    #                                            [k]
    
    # 1. Robot Block (Shared)
    Block_Robot = np.vstack([-W_A, -W_B])
    
    # 2. Payload Block (Block Diagonal)
    # Zeros must match the dimensions of the other experiment's regressor
    Zero_A = np.zeros(W_load_geom_A.shape)
    Zero_B = np.zeros(W_load_geom_B.shape)
    
    # Row A: [Geom_A, 0]
    # Row B: [0, Geom_B]
    Row_A = np.hstack([-W_load_geom_A, Zero_A]) 
    Row_B = np.hstack([Zero_B, -W_load_geom_B]) 
    Block_Payload = np.vstack([Row_A, Row_B])
    
    # 3. Scale Block
    Block_Scale = np.vstack([K_A, K_B])
    
    # 4. RHS
    RHS = np.concatenate([
        tau_payload_mass_A,                       
        tau_payload_mass_B
    ])
    
    # --- 5. Build Priors & Masks ---
    phi_r_prior = np.zeros(14 * nb_joints)
    reg_mask    = np.zeros(14 * nb_joints)
    
    for i, jname in enumerate(joint_names):
        base = i * 14
        # Inertial Params
        phi_r_prior[base+0] = phi_cad_dict.get(f"m_{jname}", 0.0)
        phi_r_prior[base+1] = phi_cad_dict.get(f"mx_{jname}", 0.0)
        phi_r_prior[base+2] = phi_cad_dict.get(f"my_{jname}", 0.0)
        phi_r_prior[base+3] = phi_cad_dict.get(f"mz_{jname}", 0.0)
        phi_r_prior[base+4] = phi_cad_dict.get(f"Ixx_{jname}", 0.0)
        phi_r_prior[base+5] = phi_cad_dict.get(f"Ixy_{jname}", 0.0)
        phi_r_prior[base+6] = phi_cad_dict.get(f"Iyy_{jname}", 0.0)
        phi_r_prior[base+7] = phi_cad_dict.get(f"Ixz_{jname}", 0.0)
        phi_r_prior[base+8] = phi_cad_dict.get(f"Iyz_{jname}", 0.0)
        phi_r_prior[base+9] = phi_cad_dict.get(f"Izz_{jname}", 0.0)
        
        # Armature Prior (Index 12)
        phi_r_prior[base+12] = armature_vals[i]
        
        # Mask Definition
        reg_mask[base+4 : base+10]    = 1.0  # Regularize Inertial
        reg_mask[base+10 : base+12] = 0.0  # Free Friction (Fv, Fs)
        reg_mask[base+12 : base+13] = 1.0  # Regularize Armature
        reg_mask[base+13 : base+14] = 0.0  # Free Offset
    
    # Helper for Load Prior
    def build_load_prior(mass, p_com, r_in, r_out, h_th):
        val_Ix = 0.5 * mass * (r_in**2 + r_out**2)
        val_Iy_Iz = (1.0/12.0) * mass * (3*(r_in**2 + r_out**2) + h_th**2)    
        I_load_at_com = np.diag([val_Ix, val_Iy_Iz, val_Iy_Iz])
        p_norm_sq = np.dot(p_com, p_com)
        p_outer   = np.outer(p_com, p_com)
        I_load_at_origin = I_load_at_com + mass * (p_norm_sq * np.eye(3) - p_outer)
        h_load = mass * p_com

        return np.array([
                h_load[0], h_load[1], h_load[2],
                I_load_at_origin[0,0], I_load_at_origin[0,1], I_load_at_origin[1,1], 
                I_load_at_origin[0,2], I_load_at_origin[1,2], I_load_at_origin[2,2]
            ])
    
    # Define Load Priors
    p_com_A = np.array([0.0775, 0.0, 0.0])
    phi_l_prior_A = build_load_prior(mass_load_A, p_com_A, 0.025/2, 0.129/2, 0.0155/2)
    
    p_com_B = np.array([0.0775, 0.0, 0.0])
    phi_l_prior_B = build_load_prior(mass_load_B, p_com_B, 0.029/2, 0.160/2, 0.023/2)
        
    # --- 6. Optimization Variables ---
    phi_robot  = cp.Variable(14 * nb_joints)
    phi_load_A = cp.Variable(9) 
    phi_load_B = cp.Variable(9)
    k_tau      = cp.Variable(nb_joints)

    # --- 7. Constraints ---
    constraints = []
    
    # A. Robot LMI
    for j, jname in enumerate(joint_names):
        base = j * 14
        m, h = phi_robot[base], phi_robot[base+1:base+4]
        I_t = cp.bmat([[phi_robot[base+4], phi_robot[base+5], phi_robot[base+7]],
                       [phi_robot[base+5], phi_robot[base+6], phi_robot[base+8]],
                       [phi_robot[base+7], phi_robot[base+8], phi_robot[base+9]]])
        
        # Physical Consistency (Mass/Inertia)
        constraints.append(cp.bmat([[0.5*cp.trace(I_t)*np.eye(3)-I_t, cp.reshape(h,(3,1))],
                                    [cp.reshape(h,(1,3)), cp.reshape(m,(1,1))]]) >> 0)
        # Positive Friction
        constraints.append(phi_robot[base+10] >= 0)
        constraints.append(phi_robot[base+11] >= 0)
        # Positive Armature
        constraints.append(phi_robot[base+12] >= 0)
        # Mesh bounds
        b_min, b_max = mesh_bounds[jname]
        constraints += [
            h[0] >= m * b_min[0], h[0] <= m * b_max[0],
            h[1] >= m * b_min[1], h[1] <= m * b_max[1],
            h[2] >= m * b_min[2], h[2] <= m * b_max[2]
        ]

    # B. Payload A LMI
    h_A = phi_load_A[0:3]
    I_A = cp.bmat([[phi_load_A[3], phi_load_A[4], phi_load_A[6]],
                   [phi_load_A[4], phi_load_A[5], phi_load_A[7]],
                   [phi_load_A[6], phi_load_A[7], phi_load_A[8]]])
    m_A_const = np.array([[mass_load_A]])
    constraints.append(cp.bmat([[0.5*cp.trace(I_A)*np.eye(3)-I_A, cp.reshape(h_A,(3,1))],
                                [cp.reshape(h_A,(1,3)), m_A_const]]) >> 0)
    
    # C. Payload B LMI
    h_B = phi_load_B[0:3]
    I_B = cp.bmat([[phi_load_B[3], phi_load_B[4], phi_load_B[6]],
                   [phi_load_B[4], phi_load_B[5], phi_load_B[7]],
                   [phi_load_B[6], phi_load_B[7], phi_load_B[8]]])
    m_B_const = np.array([[mass_load_B]])
    constraints.append(cp.bmat([[0.5*cp.trace(I_B)*np.eye(3)-I_B, cp.reshape(h_B,(3,1))],
                                [cp.reshape(h_B,(1,3)), m_B_const]]) >> 0)

    # D. Scale Bounds
    constraints.append(k_tau >= 0.0)

    # --- 8. Cost & Solve ---
    term_robot   = Block_Robot @ phi_robot
    term_payload = Block_Payload @ cp.hstack([phi_load_A, phi_load_B])
    term_scale   = Block_Scale @ k_tau
    
    pred_y = term_robot + term_payload + term_scale
    
    N_total = len(RHS)
    
    # Regularization Terms
    diff_robot = phi_robot - phi_r_prior
    weighted_diff_robot = cp.multiply(reg_mask, diff_robot)
    
    cost = (1.0/N_total) * cp.sum_squares(pred_y - RHS) + \
           lambda_r * cp.sum_squares(weighted_diff_robot) + \
           lambda_l * cp.sum_squares(phi_load_A - phi_l_prior_A) + \
           lambda_l * cp.sum_squares(phi_load_B - phi_l_prior_B)

    print(f"[SDP] Solving Dual Load ID (Mass A={mass_load_A}, Mass B={mass_load_B})...")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True, tol_gap_abs=1e-5, tol_gap_rel=1e-5, max_iter=2000)
    
    # --- 9. Retrieve Values & Costs ---
    phi_robot_val = phi_robot.value
    k_tau_val     = k_tau.value
    phi_load_A_val = phi_load_A.value
    phi_load_B_val = phi_load_B.value
    
    # Calculate Reconstruction Torques    
    # Payloads
    tau_load_A = (W_load_geom_A @ phi_load_A_val) + (W_load_mass_A * mass_load_A)
    tau_load_B = (W_load_geom_B @ phi_load_B_val) + (W_load_mass_B * mass_load_B)

    # Breakdown Costs
    term_fit_val   = (1.0/N_total) * cp.sum_squares(pred_y - RHS).value
    term_robot_val = lambda_r * cp.sum_squares(weighted_diff_robot).value
    term_load_val  = lambda_l * (cp.sum_squares(phi_load_A - phi_l_prior_A).value + \
                                 cp.sum_squares(phi_load_B - phi_l_prior_B).value)
    
    total_val = term_fit_val + term_robot_val + term_load_val

    print("-" * 60)
    print(f"{'COST TERM BREAKDOWN':<30} | {'VALUE':<15}")
    print("-" * 60)
    print(f"{'1. Data Fit (MSE)':<30} | {term_fit_val:.6e}")
    print(f"{'2. Robot Reg':<30} | {term_robot_val:.6e}")
    print(f"{'3. Payload Reg':<30} | {term_load_val:.6e}")
    print("-" * 60)
    print(f"{'TOTAL OPTIMIZED COST':<30} | {total_val:.6e}")
    print("-" * 60)

    # Return tuple of results
    return (phi_robot_val, phi_load_A_val, phi_load_B_val, k_tau_val, 
           (tau_load_A, tau_load_B),
           (term_fit_val, term_robot_val, term_load_val))