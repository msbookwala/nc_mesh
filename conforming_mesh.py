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
    N1 = 0.5 * (1 - xi)
    N2 = 0.5 * (1 + xi)
    return np.array([N1, N2])

def dN_dxi(xi, eta):
    dN_dxi = 0.25*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_deta = 0.25*np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    return np.vstack((dN_dxi, dN_deta))

def gauss_quadrature2D(func, coords, order=2):
    if order == 2:
        points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1, 1])
    else:
        raise ValueError("Unsupported order for Gauss quadrature")
    integral = 0.0
    for i in range(len(points)):
        for j in range(len(points)):
            xi = points[i]
            eta = points[j]
            weight = weights[i] * weights[j]
            integral += weight * func(xi, eta, coords)
    return integral

def elem_jacobian(xi, eta, coords):
    dNdxi = dN_dxi(xi, eta)[0, :]
    dNdeta = dN_dxi(xi, eta)[1, :]
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
    B=   dN_dxi(xi, eta).T@invJ
    return B@ B.T * detJ

def assemble_stiffness_matrix(nodes, elements):
    num_nodes = nodes.shape[0]
    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes,1)) # for now, F = 0
    for elem in elements:
        coords = nodes[elem]
        K[elem[:, None], elem] += gauss_quadrature2D(integrandK, coords)
    return K, F
def compute_lagrange_basis_matrix(points, nodes):
    N = np.zeros((len(points), len(nodes)))
    for i, point in enumerate(points):
        closest_indices = get_closest_nodes(point, nodes)
        a = np.linalg.norm(-point + nodes[closest_indices[1]]) / np.linalg.norm(nodes[closest_indices[0]] - nodes[closest_indices[1]])
        b = np.linalg.norm(point - nodes[closest_indices[0]]) / np.linalg.norm(nodes[closest_indices[0]] - nodes[closest_indices[1]])
        N[i, closest_indices[0]] = a
        N[i, closest_indices[1]] = b
    return N

def get_sampling_points(nodes, npts=2):
    # generate sampling points on gauss points along the edges of the elements
    # hard coded for 2D elements along the y edge at x = 1
    sp = np.zeros((npts*(len(nodes)-1), 2),)
    cur = 0
    if npts==2:
        for i in range(len(nodes)-1):
            for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                sp[cur, 0] = 1
                sp[cur, 1] = N_1d(xi) @ nodes[i:i+2, 1]
                cur +=1
    if npts==1:
        for i in range(len(nodes)-1):
            sp[cur, 0] = 1
            sp[cur, 1] = N_1d(0) @ nodes[i:i+2, 1]
            cur +=1
    else:
        xis = np.linspace(-1, 1, npts)
        for i in range(len(nodes)-1):
            for xi in xis:
                sp[cur, 0] = 1
                sp[cur, 1] = N_1d(xi) @ nodes[i:i+2, 1]
                cur +=1
    return sp

def compute_shepard_basis_matrix(points, nodes):
    N = np.zeros((len(points), len(nodes)))
    for i, point in enumerate(points):
        denom = 0
        wt = np.zeros(len(nodes))
        coinciding = False
        for j in range(len(nodes)):
            if np.linalg.norm(point - nodes[j]) == 0:
                coinciding = True
                ci = j
                break
            wt[j] = 1/np.linalg.norm(point - nodes[j])**2
            denom += wt[j]
            N[i, j] = wt[j]
        if coinciding:
            N[i, :] = 0
            N[i, ci] = 1
        else:
            N[i,:] /= denom
    return N

def get_closest_nodes(point, nodes):
    # get 2 closest nodes
    distances = np.linalg.norm(nodes - point, axis=1)
    closest_indices = np.argsort(distances)[:2]
    return  closest_indices


if __name__ == "__main__":

    ##############################################################################################################
    # mesh and assemble stiffness matrix for two domains
    # 1 corresponds to domain on left, 2 corresponds to domain on right
    # kappa = 1 for now
    ##############################################################################################################
    nodes1 = np.array([[0,0],
                      [1,0],
                       [2,0],
                       [0, 2/3],
                       [1, 2/3],
                       [2, 2/3],
                       [0, 4/3],
                       [1, 4/3],
                       [2, 4/3],
                       [0, 2],
                       [1, 2],
                       [2, 2]])
    elements1 = np.array([[0,1,4,3],
                          [1,2,5,4],
                          [3,4,7,6],
                          [4,5,8,7],
                          [6,7,10,9],
                          [7,8,11,10]])


    K1, F1 = assemble_stiffness_matrix(nodes1, elements1)
    # Neumann boundary conditions:
    F1[[0,3]] += -1/3
    F1[[3,6]] += -1/3
    F1[[6,9]] += -1/3

    F1[[2,5]] += 1/3
    F1[[5,8]] += 1/3
    F1[[8,11]] += 1/3

    # penalty method -> bottom right node of mesh 2 has temp = 0
    b = 1e4
    K1[2,2] += b
    F1[2] += b

    u1 = spla.spsolve(sp.csr_matrix(K1), F1)
    plt.figure()
    sc =plt.scatter(nodes1[:,0], nodes1[:,1], c=u1, cmap='viridis', s=100)
    plt.colorbar(sc)
    plt.title('Temperature Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    for i in range(len(nodes1)):
        plt.text(nodes1[i, 0], nodes1[i, 1], f'{u1[i]:.4f}', fontsize=9, ha='left', va='bottom')
    plt.show()

    # F1[[ 5]] += 1 / 3
    # u1 = spla.spsolve(sp.csr_matrix(K1), F1)
    # plt.figure()
    # sc = plt.scatter(nodes1[:, 0], nodes1[:, 1], c=u1, cmap='viridis', s=100)
    # plt.colorbar(sc)
    # plt.title('Temperature Distribution')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid()
    # for i in range(len(nodes1)):
    #     plt.text(nodes1[i, 0], nodes1[i, 1], f'{u1[i]:.4f}', fontsize=9, ha='left', va='bottom')
    # plt.show()

    exit()



