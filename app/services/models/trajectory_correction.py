import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

def detect_standing_still(seq):
    sacrum_locs = seq[:, 0, :]
    sdiff = np.diff(sacrum_locs, axis=0)
    sdiff = np.pad(sdiff, ((0,1),(0,0)), mode='edge') # repeat last value of sdiff one more time so it's the same size as seq
    sdiff = np.linalg.norm(sdiff, axis=1)
    standing_still_mask = sdiff < 0.007 # I found this threshold from watching the values of real sequences
    return standing_still_mask

def get_height_vec_estimate_AMASS(pose):
    lank = np.mean([pose[7, :], pose[10, :]], axis=0)
    rank = np.mean([pose[8, :], pose[11, :]], axis=0)
    lknee = pose[4, :]
    rknee = pose[5, :]
    height_vector = pose[0, :] - np.mean([lank, rank, lknee, rknee], axis=0)
    return height_vector

def get_hip_vec_estimate_AMASS(pose):
    return pose[2, :] - pose[1, :]

def get_perpendicular_in_dir_of_head(vec1, vec2, pose):
    res = np.cross(vec1,vec2)
    dir_of_right_toes = pose[11, :] - pose[8, :]
    dir_of_left_toes = pose[10, :] - pose[7, :]
    dir_of_head = pose[15, :] - pose[12, :]
    dir_of_knee1 = pose[5, :] - np.mean([pose[2, :], pose[8, :]], axis=0)
    dir_of_knee2 = pose[4, :] - np.mean([pose[1, :], pose[7, :]], axis=0)
    right_general_direction = np.mean([dir_of_right_toes, dir_of_left_toes, dir_of_head, dir_of_knee1, dir_of_knee2], axis=0)
    if np.dot(res, right_general_direction) < 0: res *= -1
    return res / np.linalg.norm(res) # vector of length 1


def get_rotation_matrix(vec1, vec2=np.array(([[1, 0, 0], [0, 0, 1], [0, 1, 0]])), weights=[0.8, 1, 0.6]):
    """get rotation matrix between two sets of vectors using scipy vector sets of shape N x 3"""
    r = R.align_vectors(vec2, vec1, weights=weights)
    return r[0].as_matrix()

def transform_seq_so_it_has_straight_trajectory_AMASS(seq, n_frames_est_mov_dir, window_size, polynomial):
    """ Assumes y axis roughly corressponds to height (movement in xz plane is used for trajectory reconstruction """
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, 0, :]
    
    hip_vectors    = []
    height_vectors = []
    mov_dir_ests2  = []
    for pose in centered_seq:
        hip_vector     = get_hip_vec_estimate_AMASS(pose)
        height_vector  = get_height_vec_estimate_AMASS(pose)
        mov_dir_est2   = get_perpendicular_in_dir_of_head(hip_vector, height_vector, pose)
        hip_vectors.append(hip_vector)
        height_vectors.append(height_vector)
        mov_dir_ests2.append(mov_dir_est2)
    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests2 = np.stack(mov_dir_ests2)
    
    standing_still_mask = detect_standing_still(seq)
    sacrum_locs = seq[:, 0, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0,n),(0,0)), mode='edge') # Reapeat last value n times so it's the same len as seq
    for i,m_dir_est in enumerate(m_dir):
        m_dir[i] = mov_dir_ests2[i]
        m_dir[i] = m_dir[i] / np.linalg.norm(m_dir[i])

    xhat = m_dir[:,0]
    yhat = m_dir[:,1]
    zhat = m_dir[:,2]
    if window_size is not None and polynomial is not None:
        window_size = np.min([window_size, m_dir.shape[0]])
        xhat = savgol_filter(m_dir[:,0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:,1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:,2], window_size, polynomial)

    movement_direction = np.swapaxes(np.stack([xhat, yhat, zhat]), 0, 1)
    sacrum_movements = np.diff(sacrum_locs, axis=0)
    sacrum_movements_xz = sacrum_movements[:, [0,2]]
    horizontal_movement = np.linalg.norm(sacrum_movements_xz, axis=1)

    centered_seq_rotated = []
    for i,mov_dir_vec in enumerate(movement_direction):
        rot_mat = get_rotation_matrix(np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]))
        rotated = np.stack([rot_mat.dot(joint) for joint in centered_seq[i]])
        rotated += np.array([np.sum(horizontal_movement[:i]), 0, 0])
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        centered_seq_rotated.append(rotated)
        
    centered_seq_rotated = np.stack(centered_seq_rotated)
    return centered_seq_rotated

def correct_AMASS_sequence(seq):
    seq = transform_seq_so_it_has_straight_trajectory_AMASS(seq, n_frames_est_mov_dir=15, window_size=None, polynomial=None)

    for joint in range(seq.shape[1]):
        for coord in range(3):
            seq[:, joint, coord] = gaussian_filter(seq[:, joint, coord], sigma=1.4324)
    
    return seq