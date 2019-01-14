import numpy as np

test_data = np.load('test_data_public.npz')

mocks = test_data['mocks_measurements']
dv    = test_data['dv']


r_mocks = np.dot(mocks,np.random.random(mocks.shape).T)
r_dv    = np.dot(dv,np.random.random(dv.shape).T)

np.savez('rdata_public_test.npz',tr_conf_ar= test_data['tr_conf_ar'],dv_derivatives=test_data['dv_derivatives'],
         bins_range= test_data['bins_range'],geo_bins_max = test_data['geo_bins_max'],mocks_measurements = r_mocks,
         dv = r_dv)