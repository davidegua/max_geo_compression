import numpy as np
import geometry_triangles as gt

"""
combine geometrical compression with maximal weighting:
first regroup the triangles in the optimal way;
after apply maximal compression to each subset by using the
covariance matrix estimated from the mocks for just the triangles
contained in each new bin
"""

"""
compute the covariance matrix from mocks for each subset of triangles:
return list with covariance matrices for each subset
"""
def gc_bins_cov_matrices(b0_mocks, which_bin_idx_ar, dim_new_dv):

    gc_cov_mat_list = []

    for i in range(dim_new_dv):

        bin_triangles_mocks = b0_mocks[np.where(which_bin_idx_ar==i)]

        "ensure that each bin has less triangles than half the number of mocks available"
        assert bin_triangles_mocks.shape[0] < 0.5*b0_mocks.shape[1], \
            "bin too large (%d), not enough mocks (%d)!"%(bin_triangles_mocks.shape[0],b0_mocks.shape[1])

        gc_cov_mat_list.append(np.cov(bin_triangles_mocks))

    return gc_cov_mat_list

"""
same thing for the derivatives with respect to the model
parameters --> return list of arrays
"""
def gc_bin_cov_derivatives(old_der_ar, which_bin_idx_ar,dim_new_dv):

    gc_deri_dv_list = []

    for i in range(dim_new_dv):

        bin_deri_dv = old_der_ar.T[np.where(which_bin_idx_ar==i)]
        gc_deri_dv_list.append(bin_deri_dv)

    return  gc_deri_dv_list

"""
compute the weights for each set of triangles
return a list of weights arrays
"""
def gc_bin_weights(b0_mocks, old_der_ar, which_bin_idx_ar , dim_new_dv):

    gc_cov_mat_list = gc_bins_cov_matrices(b0_mocks, which_bin_idx_ar, dim_new_dv)
    gc_deri_dv_list = gc_bin_cov_derivatives(old_der_ar, which_bin_idx_ar, dim_new_dv)

    gc_weights_list = []

    for i in range(dim_new_dv):

        if gc_cov_mat_list[i].size > ((old_der_ar.shape[0]-1)**2) :
            gc_weights = np.dot(np.linalg.inv(gc_cov_mat_list[i]),gc_deri_dv_list[i])
        if gc_cov_mat_list[i].size < (old_der_ar.shape[0]**2):
            gc_weights = np.ones(1)

        gc_weights_list.append(gc_weights)

    return gc_weights_list

"""
given the standard bispectrum return the compressed data-vector
"""
def gc_max_dv(bk0, which_bin_idx_ar, dim_new_dv, gc_weights_list):

    new_dv = []

    for i in range(dim_new_dv):

        if gc_weights_list[i].size>1:
            new_dv_el = np.dot(gc_weights_list[i].T,bk0[np.where(which_bin_idx_ar==i)])
            # "with normalisation for each bin"
            # new_dv_el = np.dot(gc_weights_list[i].T,bk0[np.where(which_bin_idx_ar==i)])/(1.*gc_weights_list[i].shape[0])

        if gc_weights_list[i].size == 1:
            new_dv_el = bk0[np.where(which_bin_idx_ar==i)]

        new_dv.append(new_dv_el)

    return np.array([item for sublist in new_dv for item in sublist])

"""
convert bispectrum mocks into compressed dv mocks
"""
def gc_max_mocks(bk0_mocks, which_bin_idx_ar, dim_gc_dv, dim_gc_max_dv, gc_weights_list):

    num_mocks = bk0_mocks.shape[1]

    gc_max_mocks_ar = np.zeros([dim_gc_max_dv, num_mocks])

    for i in range(num_mocks):

        gc_max_mocks_ar[:,i] = gc_max_dv(bk0_mocks[:,i],which_bin_idx_ar,
                                         dim_gc_dv,gc_weights_list)

    return gc_max_mocks_ar

"""
compute the covariance matrix, for jus the compressed bispectrum
and for its combination with pk and data vector
"""
def gc_max_likelihood_covs(p0_data, p2_data, b0_data, p0_mocks, p2_mocks,
                           b0_mocks, which_bin_idx_ar, dim_gc_dv,dim_gc_max_dv, gc_weights_list):


    compressed_mocks = gc_max_mocks(b0_mocks, which_bin_idx_ar,dim_gc_dv,dim_gc_max_dv,gc_weights_list)
    compressed_dv    = gc_max_dv(b0_data, which_bin_idx_ar, dim_gc_dv, gc_weights_list)

    new_p02_b0_mocks = np.append(p0_mocks,np.append(p2_mocks, compressed_mocks, axis=0), axis=0)
    new_p02_b0_data  = np.append(p0_data, np.append(p2_data, compressed_dv, axis=0), axis=0)


    gc_b0_only_cov  = np.cov(compressed_mocks)
    gc_full_cov_mat = np.cov(new_p02_b0_mocks)

    return gc_b0_only_cov, gc_full_cov_mat, new_p02_b0_data


"""
given the new bins for the data vector decided by the geometrical compression,
merge all the new bins with n_tr < n_par into one
"""
def gbin_reshape(lt, nbins_choice, par_num):

    list_of_ar, which_bin_idx_ar = gt.new_dv(lt, nbins_choice)

    temp_dim = len(list_of_ar)

    reshaped_list = []
    merged_bin    = []

    for i in range(temp_dim):
        if list_of_ar[i].shape[1]>par_num:
            reshaped_list.append(list_of_ar[i])
        if list_of_ar[i].shape[1]<=par_num:
            for j in range(list_of_ar[i].shape[1]):
                merged_bin.append(list_of_ar[i][:,j])

    merged_bin = np.array(merged_bin).T
    reshaped_list.append(merged_bin)
    reshaped_dim = len(reshaped_list)

    "fix the indexes array by replacing the triangles final destination"
    for l in range(reshaped_dim):
        # print(l,reshaped_dim)
        # print(reshaped_list[l].shape)
        for m in range(reshaped_list[l].shape[1]):
            index = find_tr_index(lt, reshaped_list[l][:,m])
            which_bin_idx_ar[index] = l

    return reshaped_list, reshaped_dim, which_bin_idx_ar



""" 
find triangle index 
"""
def find_tr_index(lt,tr):

    return np.argmin(np.sum(np.fabs(lt.T-tr),axis = 1))



