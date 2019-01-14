import numpy as np
from algo_geo_max import Best_bins


"""
given a triangle list, a bispectrum measurement acting as data-vector,
the derivatives of the model with respect to four parameters
and a set of 1400 mock meausurements, derive optimised geometrical compression
"""

test_data = np.load('rdata_public_test.npz')

tr_conf_ar         = test_data['tr_conf_ar']
dv_derivatives     = test_data['dv_derivatives']
bins_range         = test_data['bins_range']
geo_bins_max       = test_data['geo_bins_max']
mocks_measurements = test_data['mocks_measurements']
dv                 = test_data['mocks_measurements']

"save time"
bins_range = np.array([[2,4],[6,8],[3,5]])

gmax_obj = Best_bins(tr_conf_ar,dv_derivatives,bins_range,geo_bins_max)

gmax_obj.best_binnings()

gmax_obj.rearrange()

gmax_obj.merge_bin()

gmax_obj.weights(mocks_measurements)

comp_dv = gmax_obj.comp_dv(dv)

comp_mocks = gmax_obj.comp_mocks(mocks_measurements)