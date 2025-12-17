import numpy as np
import torch

def kabsch_alignment(P, Q):
    """
    Compute the optimal rotation matrix R that aligns P to Q.
    P, Q: [N, 3] sets of points.
    Returns R (3x3), t (3)
    """
    # 1. Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # 2. Center the points
    p = P - centroid_P
    q = Q - centroid_Q
    
    # 3. Compute covariance matrix
    H = np.dot(p.T, q)
    
    # 4. SVD of covariance matrix
    U, S, Vt = np.linalg.svd(H)
    
    # 5. Compute rotation
    R = np.dot(Vt.T, U.T)
    
    # 6. Special reflection case (determinant < 0)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # 7. Compute translation
    t = centroid_Q - np.dot(R, centroid_P)
    
    return R, t

def process_demonstration(wild_fingertips, wild_obj_pcd, anchor_hand, anchor_obj_centroid):
    """
    Align 'in-the-wild' demonstration to robot frame using 'in-scene' anchor.
    
    Args:
        wild_fingertips: [T, N_fingers, 3]
        wild_obj_pcd: [3D Point Cloud of Object]
        anchor_hand: [Initial hand pose of robot/anchor]
        anchor_obj_centroid: [Centroid of object in robot frame]
        
    Returns:
        aligned_fingertips, aligned_obj_pcd
    """
    # 1. Translation Alignment (Centroid matching)
    wild_obj_centroid = np.mean(wild_obj_pcd, axis=0)
    translation = anchor_obj_centroid - wild_obj_centroid
    
    wild_fingertips_t = wild_fingertips + translation
    wild_obj_pcd_t = wild_obj_pcd + translation
    
    # 2. Rotation Alignment (Kabsch on initial hand pose)
    # Extract initial hand pose (T=0)
    P_hand_0 = wild_fingertips_t[0] # [N_fingers, 3]
    Q_hand_0 = anchor_hand          # [N_fingers, 3]
    
    # We constrain rotation to Z-axis (gravity) as per paper usually, 
    # but standard Kabsch gives full 3D rotation. Paper says "Rotation around Z".
    # For full 3D alignment we use standard Kabsch.
    
    R, _ = kabsch_alignment(P_hand_0, Q_hand_0)
    
    # Apply R to all time steps
    # [T, N, 3] -> reshape to [T*N, 3] -> rotate -> reshape back
    T_steps, N_fingers, _ = wild_fingertips_t.shape
    
    flat_fingertips = wild_fingertips_t.reshape(-1, 3)
    rotated_fingertips = np.dot(flat_fingertips, R.T)
    
    flat_obj = wild_obj_pcd_t
    rotated_obj = np.dot(flat_obj, R.T)
    
    return rotated_fingertips.reshape(T_steps, N_fingers, 3), rotated_obj
    
