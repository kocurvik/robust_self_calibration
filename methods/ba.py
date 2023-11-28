import cv2
import numpy as np
import pyceres
import pycolmap
from scipy.spatial.transform import Rotation

from utils.geometry import get_K


def get_P(f_1, p_1, R=np.eye(3), t=np.zeros(3)):
    K = get_K(f_1, p_1)
    P = K @ np.column_stack([R, t])
    return P



def bundle_adjust(cam_1, cam_2, kp_1, kp_2, f_1, p_1, f_2, p_2, R, t):
    # try:
    prob = pyceres.Problem()
    loss = pyceres.HuberLoss(0.5)

    shift_1 = np.array([cam_1['width'] / 2, cam_2['height'] / 2])
    shift_2 = np.array([cam_1['width'] / 2, cam_2['height'] / 2])

    kp_1 = kp_1 + shift_1[np.newaxis, :]
    kp_2 = kp_2 + shift_2[np.newaxis, :]

    p_1 = p_1 + shift_1
    p_2 = p_2 + shift_2

    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
    q = np.array([qw, qx, qy, qz])

    P_1 = get_P(f_1, p_1)
    P_2 = get_P(f_2, p_2, R, t)

    # print(f_1)
    # print(f_2)
    # print(kp_1.shape)
    # print(kp_2.shape)

    X = cv2.triangulatePoints(P_1, P_2, kp_1.T, kp_1.T).T
    X = X[:, :3] / X[:, 3, np.newaxis]

    X = np.copy(X)

    c_1 = pycolmap.Camera(model='SIMPLE_RADIAL', width=int(cam_1['width']), height=int(cam_1['height']), params=np.array([f_1, p_1[0], p_2[0], 0.0]), id=0)
    c_2 = pycolmap.Camera(model='SIMPLE_RADIAL', width=int(cam_2['width']), height=int(cam_2['height']), params=np.array([f_2, p_2[0], p_2[0], 0.0]), id=1)

    pose_1 = pycolmap.Image(id=1, name="1", camera_id=c_1.camera_id, tvec=np.zeros(3))
    pose_2 = pycolmap.Image(id=2, name="2", camera_id=c_2.camera_id, tvec=t, qvec=q)

    step = max(1, len(kp_1) // 100)

    for i in range(0, len(kp_1), step):
        idx = np.copy(i)

        cost_1 = pyceres.factors.BundleAdjustmentCost(c_1.model_id, kp_1[idx])
        prob.add_residual_block(cost_1, loss, [pose_1.qvec, pose_1.tvec, X[idx], c_1.params])

        cost_2 = pyceres.factors.BundleAdjustmentCost(c_2.model_id, kp_2[idx])
        prob.add_residual_block(cost_2, loss, [pose_2.qvec, pose_2.tvec, X[idx], c_2.params])

    prob.set_parameterization(pose_1.qvec, pyceres.QuaternionParameterization())
    prob.set_parameter_block_constant(pose_1.qvec)
    prob.set_parameter_block_constant(pose_1.tvec)
    prob.set_parameterization(pose_2.qvec, pyceres.QuaternionParameterization())

    # print(prob.num_parameter_bocks(), prob.num_parameters(), prob.num_residual_blocks(), prob.num_residuals())
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = False
    options.num_threads = -1
    # options.max_num_iterations = 10
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    # print(summary.BriefReport())

    f_1_est = c_1.params[0]
    f_2_est = c_2.params[0]

    p_1_est = c_1.params[1:3] - shift_1
    p_2_est = c_2.params[1:3] - shift_2

    R = Rotation.from_quat(np.array([*pose_2.qvec[1:], pose_2.qvec[0]])).as_matrix()
    t = pose_2.tvec
    time = summary.total_time_in_seconds
    iter = summary.inner_iterations_used

    # except Exception:
    #     print("BA exception")
    #     return f_1, f_2, R, t, p_1 - shift_1, p_2 - shift_2, 60, 50



    return f_1_est, f_2_est, p_1_est, p_2_est, R, t, time, iter