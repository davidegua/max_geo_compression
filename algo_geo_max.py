import numpy as np
import geometry_triangles as gt
import maximal_geo_fun as mg



"""
given a list of triangles, the value of 3pt function associated and 
its derivatives --> define compression and return function to compress
data-vector and mocks (from which the covariance matrix of the compressed
data vector can be computed)
"""

class Best_bins:
    def __init__(self,tr_conf_ar, dv_derivatives, bins_range, geo_bins_max):
        self.tr_ar   = tr_conf_ar
        self.dv_drv  = dv_derivatives
        self.rbins   = bins_range
        self.ngb_max = geo_bins_max
        self.par_num = dv_derivatives.shape[0]

        print('object initialised')


    def best_binnings(self):
        self.best_bins, self.s_j = gt.best_binning(self.tr_ar,self.dv_drv,self.rbins,self.ngb_max)

        print('best bins found', self.best_bins)


    def rearrange(self):
        self.list_of_ar, self.which_b_idx_ar = gt.new_dv(self.tr_ar,self.best_bins)
        self.new_gdim = len(self.list_of_ar)

        print('map from dv to compressed dv derived')

        print("gc dimension before merging", self.new_gdim)

    "merge in single bin all bins with less triangles than number of parameters"
    def merge_bin(self):
        self.list_of_ar, self.new_gdim, self.which_b_idx_ar = mg.gbin_reshape(self.tr_ar,
                                                                              self.best_bins,
                                                                              self.par_num)

        print("gc dimension after merging", self.new_gdim)



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


