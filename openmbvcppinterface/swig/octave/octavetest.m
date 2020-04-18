addpath([getenv('OPENMBVCPPINTERFACE_PREFIX') '/bin']); % This path must point to your MBSim-Env installation 'bin' directory
OpenMBV;
global OpenMBV;
% If you get an error message that openmbv.oct can not be loaded add the above path on Windows
% to the PATH and on Linux to the LD_LIBRARY_PATH environment variable before starting octave.
% Or execute the following code before using OpenMBV in a octave session.
% savedDir=pwd;
% cd([prefix '/bin']);
% OpenMBV;
% cd(savedDir);

% main program
function main()
  global OpenMBV;

  % create main group
  group=OpenMBV.ObjectFactory.create_Group();
  group.setName('MBS');
  
  % add some objects to group in this subroutine and return a added cube
  cube=createMyGroup(group);

  % add a IndexedFaceSet
  ifs=OpenMBV.ObjectFactory.create_IndexedFaceSet();
  ifs.setName('IFS');
  indices=[3; 7; 4; 2]
  ifs.setIndices(indices);
  group.addObject(ifs);
   
  % create H5 and xml file
  group.setFileName('MBS_outfile.ombvx');
  group.write(true, true);
  
  % some action on the returned cube
  cube.getName()
  cube.getReferenceFrame()
  cube.getLength()
  cube.append([0.4 1 2 3 4 5 6 0.25]);
end


% a subroutine
function cube=createMyGroup(g)
  global OpenMBV;

  % create a cube
  cube=OpenMBV.ObjectFactory.create_Cube();
  cube.setName('Box1');
  cube.setReferenceFrame(true);
  cube.setLength(1.234);
  
  % create a cuboid
  cuboid=OpenMBV.ObjectFactory.create_Cuboid();
  cuboid.setName('Box2');
  cuboid.setReferenceFrame(true);
  cuboid.setLength(1.234, 3, 4);
  
  % add cube and cuboid to group
  g.addObject(cube);
  g.addObject(cuboid);

  % some actions on cuboid
  cuboid.getLength()
  cuboid.setLength([5.6; 9.7; 3.5]);
  cuboid.getLength()
end


main();
