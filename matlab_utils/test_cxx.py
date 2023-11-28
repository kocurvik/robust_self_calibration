import numpy as np
import iterative_focal
import matlab.engine


def random_test():
    R = np.random.rand(3, 3)
    u, s, vt = np.linalg.svd(R)
    s[2] = 0
    F = u @ np.diag(s) @ vt

    f1 = np.random.rand() * 1000 + 500
    f2 = np.random.rand() * 1000 + 500
    u1, v1, u2, v2 = np.random.rand(4) * 20

    # eng = matlab.engine.start_matlab()
    # s = eng.genpath('matlab_utils')
    # eng.addpath(s, nargout=0)
    #
    # ec1, ec2 = eng.problem_setup(matlab.double(F.tolist()), float(f1), float(u1),
    #                              float(v1), float(f2), float(u2), float(v2),
    #                              float(1.0), float(0.1), float(1.0), float(0.1),
    #                              nargout=2)


    out = iterative_focal.focals_from_f(F, f1, u1, v1, f2, u2, v2, 1.0, 0.1, 1.0, 0.1, 50, False)

    print(out[-2])
    print(out[-1])



if __name__ == '__main__':
    random_test()