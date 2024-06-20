import numpy as np
import pandas as pd

def conditional_mutual_information(X, Y, Z, data):
    xyz_joint_prob = compute_joint_prob(data, [X, Y, Z])
    xz_joint_prob = compute_joint_prob(data, [X, Z])
    yz_joint_prob = compute_joint_prob(data, [Y, Z])
    z_prob = compute_joint_prob(data, [Z])

    cmi = 0.0
    for x in xyz_joint_prob.index.levels[0]:
        for y in xyz_joint_prob.index.levels[1]:
            for z in xyz_joint_prob.index.levels[2]:
                p_xyz = xyz_joint_prob.get((x, y, z), 0)
                p_xz = xz_joint_prob.get((x, z), 0)
                p_yz = yz_joint_prob.get((y, z), 0)
                p_z = z_prob.get(z, 0)
                if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                    cmi += p_xyz * np.log(p_xyz * p_z / (p_xz * p_yz))
    return cmi


def compute_joint_prob(data, vars):
    joint_counts = pd.crosstab([data[var] for var in vars], dropna=False)
    joint_prob = joint_counts / np.sum(joint_counts.values)
    return joint_prob
