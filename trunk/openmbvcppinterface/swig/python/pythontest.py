import OpenMBV;

group=OpenMBV.Group();
group.setName("MBS");

cube=OpenMBV.Cube();
cube.setName("Box1");
cube.setReferenceFrame(True);
cube.setLength(1.234);
cube.setLength(OpenMBV.ScalarParameter(1.235));
cube.setLength(OpenMBV.ScalarParameter("p1", 1.236));

cuboid=OpenMBV.Cuboid();
cuboid.setName("Box2");
cuboid.setReferenceFrame(True);
cuboid.setLength(1.234, 3, 4);

group.addObject(cube);
group.addObject(cuboid);

group.write(True, True);

print(cube.getName());
print(cube.getReferenceFrame());
print(cube.getLength());
print(cuboid.getLength());

cuboid.setLength([5.6, 9.7, 3.5]);
print(cuboid.getLength());

cube.append([0.1, 1, 2, 3, 4, 5, 6, 0.25]);
