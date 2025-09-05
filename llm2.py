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

def gauss_quadrature1D(func, coords, order=2):
    if order != 2:
        raise ValueError("Unsupported order for Gauss quadrature")
    points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.array([1.0, 1.0])
    integral = 0.0
    for xi in points:
        integral += func(xi, coords)
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

def integrandM_1d(xi, coords):
    Nvec = N_1d(xi)
    J = np.linalg.norm(coords[1] - coords[0]) / 2
    return np.outer(Nvec, Nvec) * J

def assemble_mass_matrix_1d(nodes, elements):
    num_nodes = nodes.shape[0]
    M = np.zeros((num_nodes, num_nodes))
    for elem in elements:
        coords = nodes[elem]
        M[np.ix_(elem, elem)] += gauss_quadrature1D(integrandM_1d, coords)
    return M


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
    xis = np.linspace(-1, 1, npts+2)[1:-1]
    cur = 0
    for i in range(len(nodes)-1):
        for xi in xis:
            sp[cur, 0] = nodes[i, 0]
            sp[cur, 1] = N_1d(xi) @ nodes[i:i+2, 1]
            cur += 1
    return sp

def interface_quad(points_on_interface, npts=2):
    y = points_on_interface[:, 1]
    # 2-pt Gauss in [-1,1]
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) if npts == 2 else \
         np.linspace(-1, 1, npts+2)[1:-1]
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

    # F1[1] += 1
    # F1[5] += 1
    #
    # F2[0] += -1
    # F2[6] += -1

    b = 1e4
    T0 = 0.0
    K2[1, 1] += b
    F2[1]    += b*T0

    # K1[4,4] += b
    # F1[4]    += b*1.0


    interface_nodes1 = [1, 3, 5]
    interface_nodes2 = [0, 2, 4, 6]
    internal_nodes1  = [0, 2, 4]
    internal_nodes2  = [1, 3, 5, 7]

    num_interface_nodes = 7
    y_vals = np.linspace(0, 2, num_interface_nodes)
    interface_nodes = np.column_stack((np.ones(num_interface_nodes), y_vals))
    interface_elements = np.array([[i, i + 1] for i in range(num_interface_nodes - 1)])



    li1 = len(internal_nodes1)
    li2 = len(internal_nodes2)
    nb  = len(interface_nodes)  # frame size

    Gamma1 = compute_lagrange_basis_matrix(interface_nodes, nodes1[interface_nodes1])
    Gamma2 = compute_lagrange_basis_matrix(interface_nodes, nodes2[interface_nodes2])
    M = assemble_mass_matrix_1d(interface_nodes, interface_elements)

    B1 = M@Gamma1
    B2 = -M@Gamma2


    length = nodes1.shape[0] + nodes2.shape[0] + nb
    K_fin  = np.zeros((length, length))
    F_fin  = np.zeros((length, 1))

    K1_mapping = [0,1,2,3,4,5]
    K2_mapping = [6,7,8,9,10,11,12,13]
    lam_mapping = [14+i for i in range(nb)]

    K_fin[np.ix_(K1_mapping, K1_mapping)] += K1
    K_fin[np.ix_(K2_mapping, K2_mapping)] += K2
    F_fin[K1_mapping, :] += F1
    F_fin[K2_mapping, :] += F2

    K_fin[np.ix_([K1_mapping[i] for i in interface_nodes1], lam_mapping)] += B1.T
    K_fin[np.ix_(lam_mapping, [K1_mapping[i] for i in interface_nodes1])] += B1
    K_fin[np.ix_([K2_mapping[i] for i in interface_nodes2], lam_mapping)] += B2.T
    K_fin[np.ix_(lam_mapping, [K2_mapping[i] for i in interface_nodes2])] += B2


    # compute rank of B1 and B2
    print("Rank of B1:", np.linalg.matrix_rank(B1))
    print("Rank of B2:", np.linalg.matrix_rank(B2))
    # print(B_)




    u_fin = spla.spsolve(sp.csr_matrix(K_fin), F_fin)

    # nodes_all = np.array([[0,0],
    #                       [0,1],
    #                       [0,2],
    #                       [1,0],
    #                       [1,2/3],
    #                       [1,1],
    #                       [1,4/3],
    #                       [1,2],
    #                       [2,0],
    #                       [2,2/3],
    #                       [2,4/3],
    #                       [2,2]])
    plt.figure()
    sc1 = plt.scatter(nodes1[:, 0], nodes1[:, 1], c=u_fin[K1_mapping], s=100)
    plt.colorbar(sc1)

    plt.title(f'Temperature Distribution  (lagrange only)')

    plt.xlabel('X'); plt.ylabel('Y')
    plt.xlim(-0.2, 2.2); plt.ylim(-0.2, 2.2)
    plt.grid(True)
    for i in range(len(nodes1)):
        plt.text(nodes1[i, 0], nodes1[i, 1] + 0.05, f'{u_fin[i]:.4f}', fontsize=9,
                 ha='left', va='bottom')
    plt.show()

    plt.figure()
    sc2 = plt.scatter(nodes2[:, 0], nodes2[:, 1], c=u_fin[K2_mapping], s=100, marker='s')
    plt.colorbar(sc2)
    plt.title(f'Temperature Distribution  (lagrange only)')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.xlim(-0.2, 2.2); plt.ylim(-0.2, 2.2)
    plt.grid(True)
    for i in range(len(nodes2)):
        plt.text(nodes2[i, 0], nodes2[i, 1] + 0.05, f'{u_fin[i+6]:.4f}', fontsize=9,
                 ha='left', va='bottom')
    plt.show()
