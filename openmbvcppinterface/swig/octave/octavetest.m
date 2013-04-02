OpenMBV;

group=OpenMBV.Group();
group.setName('MBS');

cube=OpenMBV.Cube();
cube.setName('Box1');
cube.setReferenceFrame(true);
cube.setLength(1.234);

cuboid=OpenMBV.Cuboid();
cuboid.setName('Box2');
cuboid.setReferenceFrame(true);
cuboid.setLength(1.234, 3, 4);

group.addObject(cube);
group.addObject(cuboid);

cube.getName()
cube.getReferenceFrame()
cube.getLength()
cuboid.getLength()

cuboid.setLength([5.6; 9.7; 3.5]);
cuboid.getLength()

group.write(true, true);

cube.append([0.4 1 2 3 4 5 6 0.25]);
