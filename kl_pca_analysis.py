import numpy as np
import bk_fun_rsd as bk_rsd

"""""""""""""""""""""""""""""""""""""""""""""""""""""
kl analysis function for the new parameter vector
for which the fisher matrix is diagonal.
COMPUTE THE WEIGHTS AND MAKE THEM ORTHOGONAL (should already be)
"""""""""""""""""""""""""""""""""""""""""""""""""""""

def pca_kl_w_f(inv_cov, dv_ders,
               par_dim):

    print("inside PCA - KL analysis")

    inv_cov_w = inv_cov

    """ WEIGHTING VECTORS """
    w_vecs_u = np.dot(dv_ders, inv_cov_w)

    """ COMPUTE THE FISHER MATRIX """
    w_dim = w_vecs_u[0].size

    f_dim = par_dim

    kl_f_mat = np.ndarray([f_dim, f_dim], dtype=float)

    mu_i_vec = np.copy(dv_ders)

    print(mu_i_vec.shape)
    print(dv_ders.shape)
    w_vecs_u_ren = np.ndarray([f_dim,w_dim], dtype=float)
    w_vecs_m_ren = np.ndarray([f_dim,w_dim], dtype=float)

    for i in range(0, f_dim):
        for j in range(0, f_dim):

            kl_f_mat[i, j] = np.dot(mu_i_vec[i].T,np.dot(inv_cov_w,mu_i_vec[j]))

    print('fmat')
    print(kl_f_mat)

    """ DIAGONALISE THE FISHER MATRIX """
    c_arr, w_vecs_m = bk_rsd.diag_fisher_f(kl_f_mat, w_vecs_u)

    for i in range(0, f_dim):
        w_vecs_u_ren[i, :] = w_vecs_u[i, :] / np.amax(np.fabs(w_vecs_u[i, :]))
        w_vecs_m_ren[i, :] = w_vecs_m[i, :] / np.amax(np.fabs(w_vecs_m[i, :]))

    print(w_vecs_m.shape)
    print(w_dim)

    return kl_f_mat, w_vecs_m, w_vecs_u,  w_vecs_m_ren,  w_vecs_u_ren



""" BOSS monopole version """
def pca_kl_w_f_mono(inv_cov_w, dv_ders,
               par_dim):

    print(" KL weights creation")



    """ WEIGHTING VECTORS """
    w_vecs_u = np.dot(dv_ders, inv_cov_w)


    """ COMPUTE THE FISHER MATRIX """
    w_dim = w_vecs_u[0].size

    f_dim = par_dim

    kl_f_mat = np.ndarray([f_dim, f_dim], dtype=float)

    mu_i_vec = np.copy(dv_ders)

    print(mu_i_vec.shape)
    print(dv_ders.shape)
    w_vecs_u_ren = np.ndarray([f_dim,w_dim], dtype=float)
    w_vecs_m_ren = np.ndarray([f_dim,w_dim], dtype=float)

    for i in range(0, f_dim):
        for j in range(0, f_dim):

            kl_f_mat[i, j] = np.dot(mu_i_vec[i].T,np.dot(inv_cov_w,mu_i_vec[j]))

    print('fmat')
    print(kl_f_mat)

    """ DIAGONALISE THE FISHER MATRIX """
    c_arr, w_vecs_m = bk_rsd.diag_fisher_f(kl_f_mat, w_vecs_u)

    for i in range(0, f_dim):
        w_vecs_u_ren[i, :] = w_vecs_u[i, :] / np.amax(np.fabs(w_vecs_u[i, :]))
        w_vecs_m_ren[i, :] = w_vecs_m[i, :] / np.amax(np.fabs(w_vecs_m[i, :]))

    print(w_vecs_m.shape)
    print(w_dim)

    return kl_f_mat, w_vecs_m, w_vecs_u,  w_vecs_m_ren,  w_vecs_u_ren


































