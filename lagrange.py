import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def N(xi, eta):
    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)
    return np.array([N1, N2, N3, N4])


def N_1d(xi):
    return np.array([0.5*(1 - xi), 0.5*(1 + xi)])


def dN_dxi(xi, eta):
    dNdxi  = 0.25*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dNdeta = 0.25*np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    return np.vstack((dNdxi, dNdeta))


def gauss_quadrature2D(func, coords, order=2):
    if order != 2:
        raise ValueError("Unsupported order for Gauss quadrature")
    points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.array([1.0, 1.0])
    integral = 0.0
    for xi in points:
        for eta in points:
            integral += (func(xi, eta, coords))
    return integral


def elem_jacobian(xi, eta, coords):
    dNdxi, dNdeta = dN_dxi(xi, eta)[0, :], dN_dxi(xi, eta)[1, :]
    J = np.zeros((2, 2))
    J[0, 0] = dNdxi @ coords[:, 0]
    J[0, 1] = dNdxi @ coords[:, 1]
    J[1, 0] = dNdeta @ coords[:, 0]
    J[1, 1] = dNdeta @ coords[:, 1]
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    return J, detJ, invJ


def integrandK(xi, eta, coords):
    J, detJ, invJ = elem_jacobian(xi, eta, coords)
    B = dN_dxi(xi, eta).T @ invJ.T
    return (B @ B.T) * detJ


def assemble_stiffness_matrix(nodes, elements):
    num_nodes = nodes.shape[0]
    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))
    for elem in elements:
        coords = nodes[elem]
        K[elem[:, None], elem] += gauss_quadrature2D(integrandK, coords)
    return K, F


def get_closest_nodes(point, nodes):
    d = np.linalg.norm(nodes - point, axis=1)
    return np.argsort(d)[:2]


def compute_lagrange_basis_matrix(points, nodes):

    Nmat = np.zeros((len(points), len(nodes)))
    for i, p in enumerate(points):
        i0, i1 = get_closest_nodes(p, nodes)
        denom = np.linalg.norm(nodes[i0] - nodes[i1])
        a = np.linalg.norm(nodes[i1] - p) / denom
        b = np.linalg.norm(p - nodes[i0]) / denom
        Nmat[i, i0] = a
        Nmat[i, i1] = b
    return Nmat


def compute_shepard_basis_matrix(points, nodes):
    N = np.zeros((len(points), len(nodes)))
    for i, point in enumerate(points):
        # exact coincidence check
        coinc = np.where(np.all(np.isclose(nodes, point), axis=1))[0]
        if len(coinc) > 0:
            N[i, :] = 0.0
            N[i, coinc[0]] = 1.0
            continue
        w = 1.0/np.maximum(1e-14, np.linalg.norm(nodes - point, axis=1))**2
        N[i, :] = w/np.sum(w)
    return N


def get_sampling_points(nodes, npts=2):

    sp = np.zeros((npts*(len(nodes)-1), 2))
    xis = np.linspace(-1, 1, npts+2)[1:-1]  # exclude endpoints
    cur = 0
    for i in range(len(nodes)-1):
        for xi in xis:
            sp[cur, 0] = nodes[i, 0]  # vertical line -> x constant
            sp[cur, 1] = N_1d(xi) @ nodes[i:i+2, 1]
            cur += 1
    return sp

def interface_quad(points_on_interface, npts=2):
    y = points_on_interface[:, 1]
    # 2-pt Gauss in [-1,1]
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) if npts == 2 else \
         np.linspace(-1, 1, npts+2)[1:-1]
    gw = np.array([1.0, 1.0]) if npts == 2 else np.full(npts, 2.0/npts)

    Q= []
    xconst = points_on_interface[0, 0]
    for i in range(len(y)-1):
        y0, y1 = y[i], y[i+1]
        mid, half = 0.5*(y0 + y1), 0.5*(y1 - y0)
        for k in range(len(gp)):
            Q.append([xconst, mid + half*gp[k]])
    return np.array(Q)


# ========================================================================
# Main script
# ========================================================================
if __name__ == "__main__":

    nodes1 = np.array([[0,0],
                       [1,0],
                       [0,1],
                       [1,1],
                       [0,2],
                       [1,2]])
    elements1 = np.array([[0,1,3,2],
                          [2,3,5,4]])

    nodes2 = np.array([[1,0],
                       [2,0],
                       [1,2/3],
                       [2,2/3],
                       [1,4/3],
                       [2,4/3],
                       [1,2],
                       [2,2]])
    elements2 = np.array([[0,1,3,2],
                          [2,3,5,4],
                          [4,5,7,6]])

    K1, F1 = assemble_stiffness_matrix(nodes1, elements1)
    K2, F2 = assemble_stiffness_matrix(nodes2, elements2)


    F1[[0, 2]] += -0.5
    F1[[2, 4]] += -0.5
    F2[[1, 3]] +=  1/3
    F2[[3, 5]] +=  1/3
    F2[[5, 7]] +=  1/3

    b = 1e4
    T0 = 1.0
    K2[1, 1] += b
    F2[1]    += b*T0

    interface_nodes1 = [1, 3, 5]
    interface_nodes2 = [0, 2, 4, 6]
    internal_nodes1  = [0, 2, 4]
    internal_nodes2  = [1, 3, 5, 7]

    interface_nodes = np.array([[1,0],
                                [1,2/3],
                                [1,1],
                                [1,4/3],
                                [1,2]])

    n_sampling_pts = 2
    sp1  = get_sampling_points(nodes1[interface_nodes1, :], npts=n_sampling_pts)
    N1   = compute_lagrange_basis_matrix(sp1, nodes1[interface_nodes1, :])
    psi1 = compute_shepard_basis_matrix(sp1, interface_nodes)
    R1   = np.linalg.inv(N1.T @ N1) @ N1.T @ psi1

    sp2  = get_sampling_points(nodes2[interface_nodes2, :], npts=n_sampling_pts)
    N2   = compute_lagrange_basis_matrix(sp2, nodes2[interface_nodes2, :])
    psi2 = compute_shepard_basis_matrix(sp2, interface_nodes)
    R2   = np.linalg.inv(N2.T @ N2) @ N2.T @ psi2

    li1 = len(internal_nodes1)
    li2 = len(internal_nodes2)
    nb  = len(interface_nodes)  # frame size

    K_mod1 = np.zeros((li1+nb, li1+nb))
    K_mod2 = np.zeros((li2+nb, li2+nb))
    f_mod1 = np.zeros((li1+nb, 1))
    f_mod2 = np.zeros((li2+nb, 1))

    # domain 1
    K_mod1[:li1, :li1] = K1[np.ix_(internal_nodes1, internal_nodes1)]
    K_mod1[:li1, li1:] = K1[np.ix_(internal_nodes1, interface_nodes1)] @ R1
    K_mod1[li1:, :li1] = R1.T @ K1[np.ix_(interface_nodes1, internal_nodes1)]
    K_mod1[li1:, li1:] = R1.T @ K1[np.ix_(interface_nodes1, interface_nodes1)] @ R1
    f_mod1[:li1, 0]    = F1[internal_nodes1, 0]
    f_mod1[li1:, 0]    = R1.T @ F1[interface_nodes1, 0]

    # domain 2
    K_mod2[:li2, :li2] = K2[np.ix_(internal_nodes2, internal_nodes2)]
    K_mod2[:li2, li2:] = K2[np.ix_(internal_nodes2, interface_nodes2)] @ R2
    K_mod2[li2:, :li2] = R2.T @ K2[np.ix_(interface_nodes2, internal_nodes2)]
    K_mod2[li2:, li2:] = R2.T @ K2[np.ix_(interface_nodes2, interface_nodes2)] @ R2
    f_mod2[:li2, 0]    = F2[internal_nodes2, 0]
    f_mod2[li2:, 0]    = R2.T @ F2[interface_nodes2, 0]


    quad_pts = interface_quad(interface_nodes, npts=n_sampling_pts)

    length = li1 + nb + li2 + nb
    K_fin  = np.zeros((length, length))
    F_fin  = np.zeros((length, 1))

    K1_mapping = [0,1,2,3,4,5,6,7]
    K2_mapping = [8,9,10,11,3,4,5,6,7]

    K_fin[np.ix_(K1_mapping, K1_mapping)] += K_mod1
    K_fin[np.ix_(K2_mapping, K2_mapping)] += K_mod2
    F_fin[K1_mapping, :] += f_mod1
    F_fin[K2_mapping, :] += f_mod2

    B_ = np.zeros((nb, nb))
    for i in range(nb-1):
        q_pts = quad_pts[2*i:2*i+2]
        for j in [0,1]:
            q = q_pts[j]
            nodes1_ = get_closest_nodes(q, nodes1[interface_nodes1])
            nodes2_ = get_closest_nodes(q, nodes2[interface_nodes2])
            nodes1_cords = nodes1[np.array(interface_nodes1)[nodes1_]]
            nodes2_cords = nodes2[np.array(interface_nodes2)[nodes2_]]
            N_ = N_1d(np.array([-1/np.sqrt(3), 1/np.sqrt(3)]))
            N1_ = compute_lagrange_basis_matrix(q_pts, nodes1_cords)
            N2_ = compute_lagrange_basis_matrix(q_pts, nodes2_cords)
            le = np.linalg.norm(interface_nodes[i] - interface_nodes[i+1])
            B_[[i,i+1],:] += N_[[j],:].T @ ((N1_[[j],:] @ R1[nodes1_, :]) - (N2_[[j],:]@ R2[nodes2_, :])) * le * 0.5



    # print(B_)
    iface = [3,4,5,6,7]
    ilam = [12,13,14,15,16]
    K_fin[np.ix_(iface, ilam)] += B_.T
    K_fin[np.ix_(ilam, iface)] += B_

    u_fin = spla.spsolve(sp.csr_matrix(K_fin), F_fin)

    nodes_all = np.array([[0,0],
                          [0,1],
                          [0,2],
                          [1,0],
                          [1,2/3],
                          [1,1],
                          [1,4/3],
                          [1,2],
                          [2,0],
                          [2,2/3],
                          [2,4/3],
                          [2,2]])
    plt.figure()
    sc = plt.scatter(nodes_all[:, 0], nodes_all[:, 1], c=u_fin[:12], s=100)
    plt.colorbar(sc)
    plt.title(f'Temperature Distribution - {n_sampling_pts} sampling points (lagrange)')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.xlim(-0.2, 2.2); plt.ylim(-0.2, 2.2)
    plt.grid(True)
    for i in range(len(nodes_all)):
        plt.text(nodes_all[i, 0], nodes_all[i, 1] + 0.05, f'{u_fin[i]:.4f}', fontsize=9,
                 ha='left', va='bottom')
    plt.show()
