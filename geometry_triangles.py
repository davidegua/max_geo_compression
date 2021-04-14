import numpy as np



""" regroup n triangles and their bispectrum measurements under a different
binning defined using triangles properties """

def sqrt_area(l):
    pos = 2.0 * ((l[0]*l[1])**2 +(l[0]*l[2])**2 + (l[1]*l[2])**2)
    neg = l[0]**4 + l[1]**4 + l[2]**4
    return np.sqrt(pos - neg)

def cos_tmax(l):
    sl = np.sort(l,axis=1)

    return (sl[0] ** 2 + sl[1] ** 2 - sl[2] ** 2)/ (2.0 * sl[0]*sl[1])


def ratio_cos_min(l):

    sl = np.sort(l,axis=1)

    c2 =  (sl[0] ** 2 + sl[2] ** 2 - sl[1] ** 2)/ (2.0 * sl[0]*sl[2])
    c3 =  (sl[1] ** 2 + sl[2] ** 2 - sl[0] ** 2)/ (2.0 * sl[1]*sl[2])

    return c2/c3

def conv_coords(l):

    cl = np.ndarray(l.shape)

    cl[0] = sqrt_area(l)
    cl[1] = cos_tmax(l)
    cl[2] = ratio_cos_min(l)

    return cl

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


"assign triangle coordinates to new indexing as lists"
def new_dv(lt, new_pars_bins):

    ni, nj, nk = new_pars_bins
    cl = conv_coords(lt)

    H, edges = np.histogramdd(cl.T, bins=(ni,nj,nk))

    range_i = (edges[0][1:] + edges[0][:-1]) * 0.5
    range_j = (edges[1][1:] + edges[1][:-1]) * 0.5
    range_k = (edges[2][1:] + edges[2][:-1]) * 0.5

    which_bin_idx_ar = np.ndarray(cl.shape[1],dtype=int)
    for j in range(cl.shape[1]):

        idx_i = find_nearest(range_i,cl[0,j])
        idx_j = find_nearest(range_j,cl[1,j])
        idx_k = find_nearest(range_k,cl[2,j])
        temp_l = idx_i + idx_j * ni + idx_k * ni * nj

        which_bin_idx_ar[j] = temp_l

    listofar = []
    for l in range(ni*nj*nk):

        triangles = lt[:,which_bin_idx_ar == l]

        listofar.append(triangles)

    "select only the l's that contains triangles"
    list_of_ar2 = []

    it = 0
    for m in  range(ni*nj*nk):
        if listofar[m].size > 0:
            list_of_ar2.append(listofar[m])

            which_bin_idx_ar[which_bin_idx_ar == m] = it
            it = it + 1

    # print ('new_dv_dim %d  out of %d bins'%(len(list_of_ar2),ni*nj*nk))

    return list_of_ar2, which_bin_idx_ar


""" return the bispectrum mocks and data converted into the new data vector format"""
def convert_bk_data_mocks(p0_data, p2_data, b0_data, p0_mocks, p2_mocks, b0_mocks,
                          which_bin_idx_ar, dim_new_dv):

    new_b0_mocks = np.zeros([dim_new_dv, b0_mocks.shape[1]], dtype=float)

    new_b0_data  = np.zeros(dim_new_dv, dtype=float)

    for i in range(b0_data.shape[0]):

        new_b0_mocks[which_bin_idx_ar[i]] = new_b0_mocks[which_bin_idx_ar[i]] + b0_mocks[i]

        new_b0_data[which_bin_idx_ar[i]]  = new_b0_data[which_bin_idx_ar[i]] + b0_data[i]


    new_p02_b0_mocks = np.append(p0_mocks,np.append(p2_mocks, new_b0_mocks, axis=0), axis=0)
    new_p02_b0_data  = np.append(p0_data, np.append(p2_data, new_b0_data, axis=0), axis=0)

    new_cov_mat = np.cov(new_p02_b0_mocks)

    return new_p02_b0_mocks, new_p02_b0_data, new_cov_mat

""" given bk0 model convert it into new_dv """
def convert_b0_model(bk0, which_bin_idx_ar, dim_new_dv):

    new_b0_model = np.zeros(dim_new_dv)

    for i in range(bk0.size):

        new_b0_model[which_bin_idx_ar[i]] = new_b0_model[which_bin_idx_ar[i]] + bk0[i]

    return new_b0_model

def convert_dv_geo_norm(bk0, which_bin_idx_ar, dim_new_dv,list_of_ar):

    new_b0_model = np.zeros(dim_new_dv)

    for i in range(bk0.size):
        triangles_per_bin = 1. * list_of_ar[which_bin_idx_ar[i]].size

        new_b0_model[which_bin_idx_ar[i]] += bk0[i]/triangles_per_bin

    return new_b0_model


""" given der array convert it into normalised derivative """
def convert_der_ar(bk0, which_bin_idx_ar, dim_new_dv,list_triangles_per_bin):

    new_der_ar = np.zeros(dim_new_dv)

    for i in range(bk0.size):

        triangle_per_bin = float(list_triangles_per_bin[which_bin_idx_ar[i]].size)

        new_der_ar[which_bin_idx_ar[i]] = new_der_ar[which_bin_idx_ar[i]] + bk0[i]/triangle_per_bin

    return new_der_ar

"""
FIND BEST na,nc,nr
"""

""" convert derivative array"""
def g_der_arr(lt, new_pars_bins, old_der_ar, max_num_gbins):

    list_of_ar2, which_bin_idx_ar = new_dv(lt, new_pars_bins)

    par_num   = old_der_ar.shape[0]
    new_g_dim = len(list_of_ar2)

    g_der_ar  = np.zeros([par_num,new_g_dim],dtype=float)

    if new_g_dim <  max_num_gbins:

        for i in range(par_num):

            g_der_ar[i] = convert_der_ar(old_der_ar[i],which_bin_idx_ar,new_g_dim,list_of_ar2)

    return g_der_ar

""" get a single number as a proxy of the dv derivatives variations """
def s_par_for_nchoice(lt, new_pars_bins, old_der_ar, max_num_gbins):

    g_der_ar = g_der_arr(lt,new_pars_bins,old_der_ar, max_num_gbins)

    S_ij = np.sum(np.fabs(g_der_ar), axis=1)

    return S_ij

"""
 consider all the possible combinations of binnings
 and find out the most sensitive one in terms of derivative variations
"""
def best_binning(lt, old_der_ar,bins_range, max_num_gbins):

    dim_a = bins_range[0,1] - bins_range[0,0]
    dim_c = bins_range[1,1] - bins_range[1,0]
    dim_r = bins_range[2,1] - bins_range[2,0]

    num_par = old_der_ar.shape[0]

    S_ij_grid      = np.zeros([dim_a,dim_c,dim_r,num_par],dtype=float)
    S_ij_grid_norm = np.zeros([dim_a,dim_c,dim_r,num_par],dtype=float)

    for l in range(dim_a):
        for m in range(dim_c):
            for n in range(dim_r):



                temp_bin_choice = np.array([bins_range[0,0] + l,bins_range[1,0] + m, bins_range[2,0] + n])

                S_ij_grid[l,m,n] = s_par_for_nchoice(lt, temp_bin_choice, old_der_ar, max_num_gbins)

        print(l)

    for p in range(num_par):

        S_ij_grid_norm[:,:,:,p] = S_ij_grid[:,:,:,p] / np.amax(S_ij_grid[:,:,:,p])

    s_j = np.sum(S_ij_grid_norm,axis=3)

    ind = np.unravel_index(np.argmax(s_j,axis=None),s_j.shape)

    print("index maximum", ind)

    print("corresponding number of bins ", bins_range[:,0] + np.array(ind))

    return bins_range[:,0] + np.array(ind), s_j



"""
given the result of a chain compare the 1s intervals with the ones obtained from the mcmc for dk6
"""

def compare_gc_vs_mc_1Dconst(mcmc_1s, gc_list):

    gc_1s = np.sum(gc_list[:, 1:], axis=1)*0.5

    ratio = gc_1s/mcmc_1s

    return np.sum(ratio)/ratio.size, ratio











































