import numpy as np
from algo_geo_max import Best_bins
from alternatives_to_geo import Alt_best_bins

"""
given a triangle list, a bispectrum measurement acting as data-vector,
the derivatives of the model with respect to four parameters
and a set of 1400 mock meausurements, derive optimised geometrical compression
"""

test_data = np.load('der_6par.npz')
test_data = np.load('b1_b2_fg_as_om_dk2_der_05step.npz')

tr_conf_ar         = test_data['tr_conf_ar'].T
dv_derivatives     = test_data['dv_derivatives']
bins_range         = test_data['bins_range']
geo_bins_max       = test_data['geo_bins_max']
geo_bins_max       = 117
mocks_measurements = test_data['mocks_measurements']
dv                 = test_data['dv']


"save time"
# bins_range = np.array([[2,15],[10,25],[22,35]])
bins_range = np.array([[2,15],[2,15],[2,15]])

gmax_obj = Best_bins(tr_conf_ar,dv_derivatives,bins_range,geo_bins_max)

"alternatives to geometrical compression step"
# gmax_obj = Alt_best_bins(tr_conf_ar,dv_derivatives,95,geo_bins_max)

"geometrical compression only step"
# gmax_obj.best_binnings_geo()
# gmax_obj.rearrange()
# gmax_obj.comp_dv_geo(dv)

"geo-max compression first step"
gmax_obj.best_binnings_geo_max(mocks_measurements)


gmax_obj.rearrange()

gmax_obj.merge_bin()

gmax_obj.weights(mocks_measurements)

comp_dv = gmax_obj.comp_dv(dv)

comp_mocks = gmax_obj.comp_mocks(mocks_measurements)