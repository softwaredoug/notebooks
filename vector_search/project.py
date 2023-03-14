import numpy as np
import math


def project_one(v, u):
    """Project v not in subspace -> u in subspace."""
    return u * (np.dot(v, u) / np.dot(u, u))


def project(v, A):
    """Get projection of v onto row vectors in A."""
    Q, R = np.linalg.qr(A.T)
    v = v / np.linalg.norm(v)
    projection = []
    for u in Q.T:
        projection.append(project_one(v, u))
    proj = sum(projection)
    return proj / np.linalg.norm(proj)


def round_close(dot):
    # Handle annoying rounding errors
    if dot > 1.0 and dot < 1.000001:
        return 1.0
    if dot < 0.0 and dot < -0.000001:
        return 0.0
    return dot


def dot_prod(v, u):
    dot = round_close(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)))
    return dot


def test_self_projection_same_as_self():
    A = np.array([[0.24934382, 0.03642627, 0.77059246, 0.95103857, 0.77508356],
                  [0.59719113, 0.55217467, 0.60763221, 0.80458345, 0.28441416]])
    row_norms = np.linalg.norm(A, axis=1)
    A /= row_norms[:, None]

    projected0 = project(A[0], A)
    projected1 = project(A[1], A)

    self_theta_to_0 = math.acos(dot_prod(A[0], projected0))
    self_theta_to_1 = math.acos(dot_prod(A[1], projected1))
    assert self_theta_to_0 >= 0.0 and self_theta_to_0 <= 0.00001
    assert self_theta_to_1 == 0.0 and self_theta_to_1 <= 0.00001
    assert np.isclose(A[0], projected0).all()
    assert not np.isclose(A[1], projected0).all()
    assert np.isclose(A[1], projected1).all()
    assert not np.isclose(A[0], projected1).all()


def test_should_find_solution_closer_than_span_vector():
    A = np.array([[0.24934382, 0.03642627, 0.77059246, 0.95103857, 0.77508356],
                  [0.59719113, 0.55217467, 0.60763221, 0.80458345, 0.28441416]])
    v_close_to_0 = [0.24934382, 0.03642627, 0.77059246, 0.95103857, 0.8]

    projected = project(v_close_to_0, A)

    theta_to_0 = math.acos(dot_prod(A[0], v_close_to_0))
    theta_to_plane = math.acos(dot_prod(projected, v_close_to_0))
    assert theta_to_plane < theta_to_0


def test_near_orthogonal():
    A = np.array([[0, 0, 0, 1, 1],
                  [0, 0, 0, 1.1, 0.9]])
    v_close_to_orthog = [1, 0, 0, 0.01, -0.01]

    projected = project(v_close_to_orthog, A)

    theta_to_0 = math.acos(dot_prod(A[0], v_close_to_orthog))
    theta_to_1 = math.acos(dot_prod(A[1], v_close_to_orthog))
    theta_to_plane = math.acos(dot_prod(projected, v_close_to_orthog))
    assert theta_to_0 > (math.pi / 2) - 0.1
    assert theta_to_1 > (math.pi / 2) - 0.1
    assert theta_to_plane > (math.pi / 2) - 0.1


if __name__ == "__main__":
    test_self_projection_same_as_self()
    test_should_find_solution_closer_than_span_vector()
    test_near_orthogonal()
