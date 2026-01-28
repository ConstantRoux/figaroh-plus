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
    W_A, Y_A_meas, Y_A_armature, 
    W_B, Y_B_meas, Y_B_armature,
    mass_load_known,   # Scalar (kg)
    load_joint_idx,    # Index of the joint carrying the load (0-based)
    nb_joints,
    joint_names,       # List of strings ['shoulder_pitch', ...]
    phi_cad_dict,      # Dictionary of CAD values
    lambda_r=0.1,      # Regularization weight for Robot
    lambda_l=0.1,
):
    # --- 1. Data Prep ---
    Y_A_flat = Y_A_meas.flatten(order='F')
    Y_B_flat = Y_B_meas.flatten(order='F')
    nSamples_A = Y_A_meas.shape[0]
    nSamples_B = Y_B_meas.shape[0]
    
    tau_arm_A_flat = Y_A_armature.flatten(order='F')
    tau_arm_B_flat = Y_B_armature.flatten(order='F')
    
    # --- 2. Extract Payload Regressor from W_B ---
    # The payload moves exactly like the link it is attached to.
    # We grab the inertial columns (0-9) for the specified joint index.
    # Assuming 13 params per joint: [m, h(3), I(6), Fv, Fs, Off]
    
    start_col = (load_joint_idx-1) * 13
    
    # Column 0 is Mass (scaling factor for known mass)
    W_load_mass = W_B[:, start_col + 0]  
    
    # Columns 1-9 are Geometry (h_x ... I_zz)
    W_load_geom = W_B[:, start_col + 1 : start_col + 10] 
    
    # Precompute Force Vector from Known Mass (RHS term)
    tau_payload_mass = W_load_mass * mass_load_known

    # --- 3. Build Scale Regressor (Diagonal Matrices) ---
    def build_scale_matrix(Y_flat, n_samples, n_joints):
        K = np.zeros((len(Y_flat), n_joints))
        for j in range(n_joints):
            start, end = j*n_samples, (j+1)*n_samples
            K[start:end, j] = Y_flat[start:end]
        return K

    K_A = build_scale_matrix(Y_A_flat, nSamples_A, nb_joints)
    K_B = build_scale_matrix(Y_B_flat, nSamples_B, nb_joints)

    # --- 4. Stack Linear System ---
    # System: [Robot | Payload_Geom | Scales] * X = RHS
    
    # Top (Exp A): W_A * phi_r  +  0            - K_A * k = 0
    # Bot (Exp B): W_B * phi_r  +  W_geom * phi_l - K_B * k = -tau_mass
    
    Block_Robot   = np.vstack([W_A, W_B])
    Block_Payload = np.vstack([np.zeros((len(Y_A_flat), 9)), W_load_geom])
    Block_Scale   = np.vstack([-K_A, -K_B])
    
    RHS = np.concatenate([
        -tau_arm_A_flat,                       
        -tau_payload_mass - tau_arm_B_flat
    ])
    
    # --- 5. Build Priors from Dictionary ---
    phi_r_prior = np.zeros(13 * nb_joints)
    reg_mask    = np.zeros(13 * nb_joints)
    
    for i, jname in enumerate(joint_names):
        base = i * 13
        # Fill standard params if they exist in dict, else 0.0
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
        
        reg_mask[base : base+10] = 1.0 
        reg_mask[base+10 : base+13] = 0.0
    
        r_in  = 0.03/2
        r_out = 0.126/2
        h_th  = 0.015
        # p_com = np.array([0.228 + 0.0775, 0.0, 0.0])
        p_com = np.array([0.0775, 0.0, 0.0])
        
        val_Ix = 0.5 * mass_load_known * (r_in**2 + r_out**2)
        val_Iy_Iz = (1.0/12.0) * mass_load_known * (3*(r_in**2 + r_out**2) + h_th**2)    
        I_load_at_com = np.diag([val_Ix, val_Iy_Iz, val_Iy_Iz])
        p_norm_sq = np.dot(p_com, p_com)
        p_outer   = np.outer(p_com, p_com)
        I_load_at_origin = I_load_at_com + mass_load_known * (p_norm_sq * np.eye(3) - p_outer)
        h_load = mass_load_known * p_com

        phi_l_prior = np.array([
                h_load[0], 
                h_load[1], 
                h_load[2],
                I_load_at_origin[0,0], # Ixx
                I_load_at_origin[0,1], # Ixy
                I_load_at_origin[1,1], # Iyy
                I_load_at_origin[0,2], # Ixz
                I_load_at_origin[1,2], # Iyz
                I_load_at_origin[2,2]  # Izz
            ])
        
    # --- 6. Optimization Variables ---
    phi_robot = cp.Variable(13 * nb_joints)
    phi_load  = cp.Variable(9) # [mx, my, mz, Ixx...Izz]
    k_tau     = cp.Variable(nb_joints)

    # --- 7. Constraints ---
    constraints = []
    
    # A. Robot LMI
    for j in range(nb_joints):
        base = j * 13
        m, h = phi_robot[base], phi_robot[base+1:base+4]
        I_t = cp.bmat([[phi_robot[base+4], phi_robot[base+5], phi_robot[base+7]],
                       [phi_robot[base+5], phi_robot[base+6], phi_robot[base+8]],
                       [phi_robot[base+7], phi_robot[base+8], phi_robot[base+9]]])
        # LMI Condition
        constraints.append(cp.bmat([[0.5*cp.trace(I_t)*np.eye(3)-I_t, cp.reshape(h,(3,1))],
                                    [cp.reshape(h,(1,3)), cp.reshape(m,(1,1))]]) >> 0)
        # Friction Positive
        constraints.append(phi_robot[base+10] >= 0)
        constraints.append(phi_robot[base+11] >= 0)

    # B. Payload LMI (Fixed Mass)
    h_L = phi_load[0:3]
    Ixx, Ixy, Iyy, Ixz, Iyz, Izz = phi_load[3], phi_load[4], phi_load[5], phi_load[6], phi_load[7], phi_load[8]
    I_L = cp.bmat([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    
    # Inject KNOWN MASS constant into LMI
    m_L_const = np.array([[mass_load_known]])
    constraints.append(cp.bmat([[0.5*cp.trace(I_L)*np.eye(3)-I_L, cp.reshape(h_L,(3,1))],
                                [cp.reshape(h_L,(1,3)), m_L_const]]) >> 0)

    # C. Scale Bounds
    constraints.append(k_tau >= 0.0)

    # --- 8. Cost & Solve ---
    pred_y = Block_Robot @ phi_robot + Block_Payload @ phi_load + Block_Scale @ k_tau
    N_total = len(RHS)
    
    diff_robot = phi_robot - phi_r_prior
    weighted_diff_robot = cp.multiply(reg_mask, diff_robot)
    
    # Payload regularization is usually weak (allow shape to adapt)    
    cost = (1.0/N_total) * cp.sum_squares(pred_y - RHS) + \
           lambda_r * cp.sum_squares(weighted_diff_robot) + \
           lambda_l * cp.sum_squares(phi_load - phi_l_prior)

    print(f"[SDP] Solving Calib: Robot + Scale + Payload (Mass={mass_load_known}kg)...")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    
    phi_load_val = phi_load.value
    tau_geom = W_load_geom @ phi_load_val
    tau_mass = W_load_mass * mass_load_known
    tau_load_B = tau_geom + tau_mass
    
    # --- Retrieve Values ---
    phi_robot_val = phi_robot.value
    phi_load_val  = phi_load.value
    k_tau_val     = k_tau.value
    term_fit   = (1.0/N_total) * cp.sum_squares(pred_y - RHS).value
    term_robot = lambda_r * cp.sum_squares(phi_robot - phi_r_prior).value
    term_load  = lambda_l * cp.sum_squares(phi_load - phi_l_prior).value
    total_val = term_fit + term_robot + term_load
    print("-" * 60)
    print(f"{'COST TERM BREAKDOWN':<30} | {'VALUE':<15}")
    print("-" * 60)
    print(f"{'1. Data Fit (MSE)':<30} | {term_fit:.6e}")
    print(f"{'2. Robot Reg (CAD)':<30} | {term_robot:.6e}")
    print(f"{'3. Payload Reg (Prior)':<30} | {term_load:.6e}")
    print("-" * 60)
    print(f"{'TOTAL OPTIMIZED COST':<30} | {total_val:.6e}")
    print("-" * 60)

    return phi_robot.value, phi_load.value, k_tau.value, tau_load_B