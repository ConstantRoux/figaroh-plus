# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parameter management utilities for robot identification.

This module handles parameter extraction, reordering, and management including:
- Inertial parameter extraction and reordering
- Standard additional parameters (friction, actuator inertia, offsets)
- Custom parameter support
- Parameter information queries
"""

import numpy as np


# Export public API
__all__ = [
    "reorder_inertial_parameters",
    "add_standard_additional_parameters",
    "add_custom_parameters",
    "get_standard_parameters",
    "get_parameter_info",
]


def reorder_inertial_parameters(pinocchio_params):
    """Reorder inertial parameters from Pinocchio format to desired format.

    Args:
        pinocchio_params: Parameters in Pinocchio order
            [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

    Returns:
        list: Parameters in desired order
            [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m]
    """
    if len(pinocchio_params) != 10:
        raise ValueError(
            f"Expected 10 inertial parameters, got {len(pinocchio_params)}"
        )

    # Mapping from Pinocchio indices to desired indices
    reordered = np.zeros_like(pinocchio_params)
    reordered[0] = pinocchio_params[4]  # Ixx
    reordered[1] = pinocchio_params[5]  # Ixy
    reordered[2] = pinocchio_params[7]  # Ixz
    reordered[3] = pinocchio_params[6]  # Iyy
    reordered[4] = pinocchio_params[8]  # Iyz
    reordered[5] = pinocchio_params[9]  # Izz
    reordered[6] = pinocchio_params[1]  # mx
    reordered[7] = pinocchio_params[2]  # my
    reordered[8] = pinocchio_params[3]  # mz
    reordered[9] = pinocchio_params[0]  # m

    return reordered.tolist()


def add_standard_additional_parameters(phi, params, identif_config, model):
    """Add standard additional parameters (actuator inertia, friction,
    offsets).

    Args:
        phi: Current parameter values list
        params: Current parameter names list
        identif_config: Configuration dictionary
        model: Robot model

    Returns:
        tuple: Updated (phi, params) lists
    """

    # Standard additional parameters configuration
    additional_params = [
        {
            "name": "fv",
            "enabled_key": "has_friction",
            "values_key": "fv",
            "default": 0.0,
            "description": "viscous friction",
        },
        {
            "name": "fs",
            "enabled_key": "has_friction",
            "values_key": "fs",
            "default": 0.0,
            "description": "static friction",
        },
        {
            "name": "Ia",
            "enabled_key": "has_actuator_inertia",
            "values_key": "Ia",
            "default": 0.0,
            "description": "actuator inertia",
        },
        {
            "name": "off",
            "enabled_key": "has_joint_offset",
            "values_key": "off",
            "default": 0.0,
            "description": "joint offset",
        },
    ]

    for param_def in additional_params:
        for link_idx, jname in enumerate(model.names[1:]):  # Skip world link
            param_name = f"{param_def['name']}_{jname}"

            # Get parameter value
            if identif_config.get(param_def['enabled_key'], False):
                params.append(param_name)
                try:
                    values_list = identif_config.get(
                        param_def['values_key'], []
                    )
                    if len(values_list) >= link_idx + 1:
                        value = values_list[link_idx]
                    else:
                        value = param_def['default']
                        print(
                            f"Warning: Missing {param_def['description']} "
                            f"for joint {jname}, using default: {value}"
                        )
                except (KeyError, IndexError, TypeError) as e:
                    value = param_def['default']
                    print(
                        f"Warning: Error getting {param_def['description']} "
                        f"for joint {jname}: {e}, using default: {value}"
                    )
            else:
                value = param_def['default']

            phi.append(value)

    return phi, params


def add_custom_parameters(phi, params, custom_params, model):
    """Add custom user-defined parameters.

    Args:
        phi: Current parameter values list
        params: Current parameter names list
        custom_params: Custom parameter definitions
        model: Robot model

    Returns:
        tuple: Updated (phi, params) lists
    """

    for param_name, param_def in custom_params.items():
        if not isinstance(param_def, dict):
            print(
                f"Warning: Invalid custom parameter definition for "
                f"'{param_name}', skipping"
            )
            continue

        values = param_def.get('values', [])
        per_joint = param_def.get('per_joint', True)
        default_value = param_def.get('default', 0.0)

        if per_joint:
            # Add parameter for each joint
            for link_idx, jname in enumerate(model.names[1:]):
                param_full_name = f"{param_name}_{jname}"
                params.append(param_full_name)

                try:
                    if len(values) >= link_idx + 1:
                        value = values[link_idx]
                    else:
                        value = default_value
                        # Only warn if values were provided but insufficient
                        if values:
                            print(
                                f"Warning: Missing value for custom "
                                f"parameter '{param_name}' joint "
                                f"{jname}, using default: {value}"
                            )
                except (IndexError, TypeError):
                    value = default_value
                    print(
                        f"Warning: Error accessing custom parameter "
                        f"'{param_name}' for joint {jname}, "
                        f"using default: {value}"
                    )

                phi.append(value)
        else:
            # Global parameter (not per joint)
            params.append(param_name)
            try:
                value = values[0] if values else default_value
            except (IndexError, TypeError):
                value = default_value
                print(
                    f"Warning: Error accessing global custom parameter "
                    f"'{param_name}', using default: {value}"
                )

            phi.append(value)

    return phi, params


def get_standard_parameters(
    model, identif_config=None, include_additional=True, custom_params=None
):
    """
    Get standard inertial parameters aligned PERFECTLY with the Regressor 
    column structure (Joint-Interleaved), using explicit joint names.
    """
    if identif_config is None:
        identif_config = {}
    if custom_params is None:
        custom_params = {}

    phi = []
    params = []

    inertial_params_names = [
        "m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz",
    ]

    has_friction = identif_config.get('has_friction', False)
    has_actuator_inertia = identif_config.get('has_actuator_inertia', False)
    has_joint_offset = identif_config.get('has_joint_offset', False)
    
    # 1. Retrieve the list of active joints from config
    #    Make sure this list MATCHES the order of columns in your regressor!
    active_joints = identif_config.get("active_joints", [])
    
    # Fallback if config is missing (safety)
    if not active_joints:
        print("[Warning] 'active_joints' not found in config. Falling back to all model joints.")
        active_joints = list(model.names[1:])

    # 2. Iterate strictly over the user-defined active joints
    for i, jname in enumerate(active_joints):
        
        # --- KEY FIX: Use Pinocchio to resolve the ID ---
        if not model.existJointName(jname):
            raise ValueError(f"Joint '{jname}' found in config but does not exist in Pinocchio model!")
            
        joint_id = model.getJointId(jname)
        
        # --- A. INERTIAL PARAMETERS (10) ---
        # Note: Pinocchio stores inertia in model.inertias[joint_id]
        pinocchio_params = model.inertias[joint_id].toDynamicParameters()

        for param_name in inertial_params_names:
            params.append(f"{param_name}_{jname}")
        phi.extend(pinocchio_params)

        if include_additional:
            # Note: For friction/offset lists, we assume they are ordered 
            # consistently with 'active_joints'.
            # It's safer to rely on 'i' (loop index) for these lists 
            # if they were built corresponding to active_joints.
            
            # 1. Friction (fv, fs)
            if has_friction:
                fv_list = identif_config.get('fv', [])
                fs_list = identif_config.get('fs', [])
                fv_val = fv_list[i] if len(fv_list) > i else 0.0
                fs_val = fs_list[i] if len(fs_list) > i else 0.0
                
                params.append(f"fv_{jname}")
                phi.append(fv_val)
                params.append(f"fs_{jname}")
                phi.append(fs_val)

            # 2. Actuator Inertia (ia)
            if has_actuator_inertia:
                ia_list = identif_config.get('Ia', [])
                ia_val = ia_list[i] if len(ia_list) > i else 0.0
                
                params.append(f"ia_{jname}")
                phi.append(ia_val)

            # 3. Joint Offset (off)
            if has_joint_offset:
                off_list = identif_config.get('off', [])
                off_val = off_list[i] if len(off_list) > i else 0.0
                
                params.append(f"off_{jname}")
                phi.append(off_val)

    if custom_params:
        phi, params = add_custom_parameters(phi, params, custom_params, model)

    return dict(zip(params, phi))


def get_parameter_info():
    """Get information about available parameter types.

    Returns:
        dict: Information about standard and custom parameter types
    """
    return {
        "inertial_parameters": [
            # "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz",
            # "mx", "my", "mz", "m"
            "m",
            "mx",
            "my",
            "mz",
            "Ixx",
            "Ixy",
            "Iyy",
            "Ixz",
            "Iyz",
            "Izz",
        ],
        "standard_additional": {
            "viscous_friction": {
                "name": "fv",
                "enabled_key": "has_friction",
                "values_key": "fv",
                "description": "Viscous friction coefficient",
            },
            "static_friction": {
                "name": "fs",
                "enabled_key": "has_friction",
                "values_key": "fs",
                "description": "Static friction coefficient",
            },
            "actuator_inertia": {
                "name": "Ia",
                "enabled_key": "has_actuator_inertia",
                "values_key": "Ia",
                "description": "Actuator/rotor inertia",
            },
            "joint_offset": {
                "name": "off",
                "enabled_key": "has_joint_offset",
                "values_key": "off",
                "description": "Joint position offset",
            },
        },
        "custom_parameters_format": {
            "parameter_name": {
                "values": "list of values",
                "per_joint": "boolean - if True, creates param for each joint",
                "default": "default value if not enough values provided",
            }
        },
    }
