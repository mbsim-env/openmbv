javaaddpath([getenv('OPENMBVCPPINTERFACE_PREFIX') 'bin/openmbv.jar']); % This must point to your openmbv.jar file in your MBSim installation path

group=de.mbsim_env.openmbv.Group();
group.setName('MBS');

cube=de.mbsim_env.openmbv.Cube();
cube.setName('Box1');
cube.setReferenceFrame(true);
cube.setLength(1.234);

cuboid=de.mbsim_env.openmbv.Cuboid();
cuboid.setName('Box2');
cuboid.setReferenceFrame(true);
cuboid.setLength(1.234, 3, 4);

group.addObject(cube);
group.addObject(cuboid);

group.setFileName('MBS_outfile.ombv.xml');
group.write(true, true);

cube.getName()
cube.getReferenceFrame()
cube.getLength()
cuboid.getLength()

cuboid.setLength([5.6; 9.7; 3.5]);
cuboid.getLength()

cube.append([0.4 1 2 3 4 5 6 0.25]);
