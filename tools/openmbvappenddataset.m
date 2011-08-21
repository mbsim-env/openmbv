function openmbvappenddataset(hdf5filename, path, data, chunk, h5import)
%NOTE: openmbvappenddataset was renamed to hdf5serieappenddataset
%      and moved from the OpenMBV project to the HDF5Serie project (http://hdf5serie.berlios.de).
%      Additionally you have to append '/data' to the path argument if you switch from using
%      openmbvappenddataset to hdf5serieappenddataset.
printf([
'NOTE: openmbvappenddataset was renamed to hdf5serieappenddataset\n',...
'      and moved from the OpenMBV project to the HDF5Serie project (http://hdf5serie.berlios.de).\n',...
'      Additionally you have to append "/data" to the path argument if you switch from using\n',...
'      openmbvappenddataset to hdf5serieappenddataset.\n'
]);

