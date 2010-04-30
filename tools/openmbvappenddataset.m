function openmbvappenddataset(hdf5filename, path, data, chunk, h5import)
%Usage: openmbvappenddataset(hdf5filename, path, data [, chunk [, h5import]])
%
%hdffilename: file name of the HDF5 file
%path:        path of the dataset in the HDF5 file
%data:        data in matrix from
%chunk:       [optional] chunk size (default: 10)
%h5import:    [optional] h5import command including full path (defaualt: h5import)
%             The h5import command is search in the Octave/Matlab path.
%
%Append the 2D dataset <data> to the HDF5 file <hdf5filename> at the HDF5
%path specified by <path>. The delimiter for the path is '/'. <path> must
%represent the successor route from the corrosponding root XML element
%'<Group>' over all succesor XML '<Group>' elements to the XML '<Object>'
%element.
%
%Example:
%
%Full Valid XML-File: (TS.ombv.xml)
%
%<?xml version="1.0" encoding="UTF-8"?>
%<Group name="TS" expand="true" enable="true" xmlns="http://openmbv.berlios.de/OpenMBV">
%  <Group name="Hauptgruppe1" expand="true" enable="true">
%    <Cuboid name="Box1" enable="true">
%      <minimalColorValue>0</minimalColorValue>
%      <maximalColorValue>1</maximalColorValue>
%      <staticColor>nan</staticColor>
%      <initialTranslation>[0;0;0]</initialTranslation>
%      <initialRotation>[0;0;0]</initialRotation>
%      <scaleFactor>1</scaleFactor>
%      <length>[0.5;0.5;0.5]</length>
%    </Cuboid>
%    <Frame name="P1" enable="true">
%      <minimalColorValue>0</minimalColorValue>
%      <maximalColorValue>1</maximalColorValue>
%      <staticColor>nan</staticColor>
%      <initialTranslation>[0;0;0]</initialTranslation>
%      <initialRotation>[0;0;0]</initialRotation>
%      <scaleFactor>1</scaleFactor>
%      <size>0.5</size>
%      <offset>1</offset>
%    </Frame>
%  </Group>
%</Group>
%
%Octave/Matlab commands to generate the corrosponding HDF5 file (TS.ombv.h5)
%
%openmbvappenddataset('TS.ombv.h5', 'Hauptgruppe1/Box1', [0.0 0 0 0 0 0 0 0; 0.1 1 1 1 0.1 0 0 0]);
%openmbvappenddataset('TS.ombv.h5', 'Hauptgruppe1/P1', [0.0 0 0 0 0 0 0 0; 0.1 1 1 1 0.1 0 0 0]);

% parameters
if exist('h5import')==1
  H5IMPORT=h5import;
else
  H5IMPORT='h5import';
end
if exist('chunk')==1
  CHUNK=chunk;
else
  CHUNK=10;
end
if path(1)=='/'
  path=path(2:end);
end
if size(data,1)<CHUNK
  CHUNK=size(data,1);
end

% config file for h5import
configfilename=tempname();
configfile=fopen(configfilename, 'w');
fprintf(configfile, 'PATH %s/data\n', path);
fprintf(configfile, 'INPUT-CLASS TEXTFP\n');
fprintf(configfile, 'INPUT-SIZE 64\n');
fprintf(configfile, 'RANK 2\n');
fprintf(configfile, 'DIMENSION-SIZES %d %d\n', size(data,1), size(data,2));
fprintf(configfile, 'CHUNKED-DIMENSION-SIZES %d %d\n', CHUNK, size(data,2));
fprintf(configfile, 'MAXIMUM-DIMENSIONS -1 %d\n', size(data,2));
fprintf(configfile, 'OUTPUT-CLASS FP\n');
fprintf(configfile, 'OUTPUT-SIZE 64\n');
fclose(configfile);

% data file for h5import
datafilename=tempname();
datafile=fopen(datafilename, 'w');
fprintf(datafile, '%.17f ', data');
fclose(datafile);

% append to hdf5 file using h5import
system([H5IMPORT ' ' datafilename ' -c ' configfilename ' -o ' hdf5filename]);

% delete temp files
delete(configfilename);
delete(datafilename);
