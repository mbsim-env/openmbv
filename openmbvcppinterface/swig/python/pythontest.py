import OpenMBV;

group=OpenMBV.Group();
group.setName("MBS");

cube=OpenMBV.Cube();
cube.setName("Box1");
cube.setReferenceFrame(True);
cube.setLength(1.234);

cuboid=OpenMBV.Cuboid();
cuboid.setName("Box2");
cuboid.setReferenceFrame(True);
cuboid.setLength(1.234, 3, 4);

group.addObject(cube);
group.addObject(cuboid);

print(cube.getName());
print(cube.getReferenceFrame());
print(cube.getLength());
print(cuboid.getLength());

cuboid.setLength([5.6, 9.7, 3.5]);
print(cuboid.getLength());

group.write(True, True);

cube.append([0.1, 1, 2, 3, 4, 5, 6, 0.25]);
