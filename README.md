# max_geo_compression
code for the compression of 3pt statistics

This code is based on the paper: link

For a generic 3pt statistics derive a compressed version of the data-vector.

Input quantities:

1) tr_conf_ar : 2D array containing all the triangle configurations (3 sides) used to compute the original 3pt statistics data-vector

2)dv_derivatives : 2D array containing the derivatives of the data-vector with respect the model parameters

3)bins_range: range of the number of bins for each variable that the algorithm scan looking for the best settings

4)geo_bins_max: maximum number of bins that the geometrical compression algorithm is allowed to form 
(multiply by the number of model parameters to have an upper bound limit of the compressed data-vector's dimension)

5)mocks_measurements: 2D array containing the measurements of the 3pt statistics for all the simulations

6)dv: 3pt statistics data-vector (model or measurement)

Function of the class best bins:

1) best_binnings() : find the optimal number of bins for each variable according to the geometrical compression algorithm

2) rearrange() : find how to map from the original 3pt data-vector to the geometrical compression bins

3) merge_bin() : if some of the bins contain less triangle configurations than the number of model parameters, this bins are 
                 merged into a unique one before applying maximal compression.
                 
4) weights(mocks_measurements) : find the weights for each bin given by the maximal compression algorithm

5) compp_dv(dv) : convert the 3pt statistics data-vector (model or measurement) to its compressed version

6) comp_mocks(mocks_measurements) : convert the mocks to their compressed form and return also the compressed covariance matrix

