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
    B=   dN_dxi(xi, eta).T@invJ.T
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
    # if npts==2:
    #     for i in range(len(nodes)-1):
    #         for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
    #             sp[cur, 0] = 1
    #             sp[cur, 1] = N_1d(xi) @ nodes[i:i+2, 1]
    #             cur +=1
    xis = np.linspace(-1, 1, npts+2)[1:-1]  # exclude endpoints
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

    # Neumann boundary conditions:
    F1[[0,2]] += -1/2
    F1[[2,4]] += -1/2

    F2[[1,3]] += 1/3
    F2[[3,5]] += 1/3
    F2[[5,7]] += 1/3

    F1[[1,3]] += 1/2
    F1[[3,5]] += 1/2

    F2[[0,2]] += -1/3
    F2[[2,4]] += -1/3
    F2[[4,6]] += -1/3


    # penalty method -> bottom right node of mesh 2 has temp = 0
    b = 1e4

    K2[1,1] += b
    F2[1] += b


    # hard coded the boundary and internal nodes for the example
    interface_nodes1 = [1, 3, 5]
    interface_nodes2 = [0, 2, 4, 6]
    internal_nodes1 = [0, 2, 4]
    internal_nodes2 = [1, 3, 5, 7]
    # interface nodes combined from both meshes
    interface_nodes = np.array([[1,0], [1,2/3], [1,1], [1,4/3], [1,2]])

    ##############################################################################################################
    # Generate N and Psi matrices for both domains
    ##############################################################################################################
    n_sampling_pts = 2

    sampling_points1 = get_sampling_points(nodes1[interface_nodes1, :], npts=n_sampling_pts)
    N1 = compute_lagrange_basis_matrix(sampling_points1, nodes1[interface_nodes1, :])
    psi1 = compute_shepard_basis_matrix(sampling_points1, interface_nodes)
    N1_plus = np.linalg.inv(N1.T @ N1) @ N1.T
    R1 = N1_plus @ psi1

    sampling_points2 = get_sampling_points(nodes2[interface_nodes2, :], npts=n_sampling_pts)
    N2 = compute_lagrange_basis_matrix(sampling_points2, nodes2[interface_nodes2, :])
    psi2 = compute_shepard_basis_matrix(sampling_points2, interface_nodes)
    N2_plus = np.linalg.inv(N2.T @ N2) @ N2.T
    R2  = N2_plus @ psi2

    Tf1 = np.array([[-1],[-1],[-1],[-1],[-1]])

    T_i1 = np.array([[-1],[-1],[-1]])  # internal nodes of domain 1
    RTf1 = R1 @ Tf1


    #############################################################################################################
    # Assemble the modified stiffness matrices and force vectors for both domains
    #############################################################################################################
    K_mod1 = np.zeros((len(internal_nodes1)+ len(interface_nodes), len(internal_nodes1)+ len(interface_nodes)))
    K_mod2 = np.zeros((len(internal_nodes2)+ len(interface_nodes), len(internal_nodes2)+ len(interface_nodes)))
    f_mod1 = np.zeros((len(internal_nodes1)+ len(interface_nodes), 1))
    f_mod2 = np.zeros((len(internal_nodes2)+ len(interface_nodes), 1))

    li1 = len(internal_nodes1)
    #reorder -> internal nodes first, then boundary nodes
    K_mod1[:li1, :li1] = K1[internal_nodes1,:][:, internal_nodes1]
    K_mod1[:li1, li1:] = K1[internal_nodes1,:][:, interface_nodes1] @ R1
    K_mod1[li1:, :li1] = R1.T @ K1[interface_nodes1, :][:, internal_nodes1]
    K_mod1[li1:, li1:] = R1.T @ K1[interface_nodes1, :][:, interface_nodes1] @ R1
    f_mod1[:li1, 0] = F1[internal_nodes1, 0]
    f_mod1[li1:, 0] = R1.T @ F1[interface_nodes1, 0]


    li2 = len(internal_nodes2)
    #reorder -> internal nodes first, then boundary nodes
    K_mod2[:li2, :li2] = K2[internal_nodes2,:][:, internal_nodes2]
    K_mod2[:li2, li2:] = K2[internal_nodes2,:][:, interface_nodes2] @ R2
    K_mod2[li2:, :li2] = R2.T @ K2[interface_nodes2, :][:, internal_nodes2]
    K_mod2[li2:, li2:] = R2.T @ K2[interface_nodes2, :][:, interface_nodes2] @ R2
    f_mod2[:li2, 0] = F2[internal_nodes2, 0]
    f_mod2[li2:, 0] = R2.T @ F2[interface_nodes2, 0]

    #########################################################################################################################
    # Assemble the final global stiffness matrix and force vector
    #########################################################################################################################
    length = len(internal_nodes1)+ len(interface_nodes) + len(internal_nodes2)
    K_fin = np.zeros((length, length))
    F_fin = np.zeros((length,1))

    # ordering in the final matrix:
    # internal nodes of domain 1 (left edge)-> interface nodes () -> internal nodes of domain 2 (right edge)

    K1_mapping = [0,1,2,3,4,5,6,7]
    K2_mapping = [8,9,10,11,3,4,5,6,7]

    K_fin[np.ix_(K1_mapping, K1_mapping)] +=  K_mod1
    K_fin[np.ix_(K2_mapping, K2_mapping)] +=  K_mod2

    F_fin[K1_mapping,] += f_mod1
    F_fin[K2_mapping,] += f_mod2

    u_fin = spla.spsolve(sp.csr_matrix(K_fin), F_fin)

    ###########################################################################################################################
    # Plotting the results
    ###########################################################################################################################

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
    sc =plt.scatter(nodes_all[:,0], nodes_all[:,1], c=u_fin, s=100)
    plt.colorbar(sc)
    plt.title(f'Temperature Distribution -{n_sampling_pts} sampling points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-0.2,2.2)
    plt.ylim(-0.2, 2.2)
    plt.grid()
    for i in range(len(nodes_all)):
        plt.text(nodes_all[i, 0], nodes_all[i, 1]+0.05, f'{u_fin[i]:.4f}', fontsize=9, ha='left', va='bottom')

    plt.show()






