import numpy as np
import geometry_triangles as gt
import maximal_geo_fun as mg
"""
define alternative ways to group together the triangles 
before the maximal compression step 
"""


"use ~30 out of 2734 triangles as reference for the bins"
class Alt_best_bins:
    def __init__(self,tr_conf_ar, dv_derivatives, step, geo_bins_max):
        self.tr_ar   = tr_conf_ar
        self.dv_drv  = dv_derivatives
        self.step   = step
        self.ngb_max = geo_bins_max
        self.par_num = dv_derivatives.shape[0]

        print('object initialised')


    def best_binnings(self):
        self.ref_triangles = self.tr_ar[:,::self.step]

        print(" %d ref triangles selected, step: %d"%(self.ref_triangles.shape[1],self.step))


    def rearrange(self):
        "two methods of defining the bins before the maximal compression step"
        "using reference triangles to define the bin"
        # self.list_of_ar, self.which_b_idx_ar = alt_new_dv_reference(self.tr_ar,self.ref_triangles)
        "randomly binning together the triangles"
        self.list_of_ar, self.which_b_idx_ar = alt_new_dv_random(self.tr_ar,int(self.ngb_max))
        self.new_gdim = len(self.list_of_ar)

        print('map from dv to compressed dv derived')

        print("gc dimension before merging", self.new_gdim)


    "obtain the weigths using kl compression for each new bin"
    def weights(self,mocks_measurements):
        self.gc_weights_list = mg.gc_bin_weights(mocks_measurements,self.dv_drv,
                                                 self.which_b_idx_ar,self.new_gdim)
        self.dim_cdv_max = len(self.gc_weights_list) * self.par_num

        print('weights derived for the maximal compression of each bin')
        print('final compressed dimension ',self.dim_cdv_max)

    "convert 3pt statistic dv into compressed dv"
    def comp_dv(self,dv):
        return mg.gc_max_dv(dv,self.which_b_idx_ar,self.new_gdim,self.gc_weights_list)


    "convert 3pt mocks measurements into compressed mocks"
    def comp_mocks(self,mock_measurements):
        comp_mocks = mg.gc_max_mocks(mock_measurements,self.which_b_idx_ar,
                                     self.new_gdim,self.dim_cdv_max,self.gc_weights_list)

        return comp_mocks, np.cov(comp_mocks)



"assign triangle coordinates to new indexing as lists based on reference triangles"
def alt_new_dv_reference(lt, ref_tr):
    dim2 = lt.shape[1]
    nref = ref_tr.shape[1]

    which_bin_idx_ar = np.ndarray(dim2,dtype=int)

    for i in range(dim2):
        k_diff_vec = np.sum(np.fabs(ref_tr.T - lt[:,i]), axis=1)
        which_bin_idx_ar[i] = np.argmin(k_diff_vec)

    listofar = []
    for l in range(nref):

        triangles = lt[:,which_bin_idx_ar == l]

        listofar.append(triangles)

    # print(len(listofar),np.unique(which_bin_idx_ar),np.unique(which_bin_idx_ar).size)

    "select only the l's that contains triangles"
    list_of_ar2 = []

    it = 0
    for m in  range(nref):
        if listofar[m].size > 0:
            list_of_ar2.append(listofar[m])

            which_bin_idx_ar[which_bin_idx_ar == m] = it
            it = it + 1

    # print ('new_dv_dim %d  out of %d bins'%(len(list_of_ar2),ni*nj*nk))

    return list_of_ar2, which_bin_idx_ar

"assign triangle coordinates to new indexing as lists, randomly"
def alt_new_dv_random(lt, nbins):
    dim2 = lt.shape[1]


    which_bin_idx_ar = np.random.randint(0,nbins-1,dim2)

    listofar = []
    for l in range(nbins):

        triangles = lt[:,which_bin_idx_ar == l]

        listofar.append(triangles)

    # print(len(listofar),np.unique(which_bin_idx_ar),np.unique(which_bin_idx_ar).size)

    "select only the l's that contains triangles"
    list_of_ar2 = []

    it = 0
    for m in  range(nbins):
        if listofar[m].size > 0:
            list_of_ar2.append(listofar[m])

            which_bin_idx_ar[which_bin_idx_ar == m] = it
            it = it + 1

    # print ('new_dv_dim %d  out of %d bins'%(len(list_of_ar2),ni*nj*nk))

    return list_of_ar2, which_bin_idx_ar
