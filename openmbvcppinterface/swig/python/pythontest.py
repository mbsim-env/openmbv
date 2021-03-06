import sys
import os

sys.path.append(os.environ['OPENMBVCPPINTERFACE_PREFIX']+"/bin") # This path must point to your MBSim-Env installation 'bin' directory

import OpenMBV

# main program
def main():
  # create main group
  group=OpenMBV.ObjectFactory.create_Group()
  group.setName("MBS")
  
  # add some objects to group in this subroutine and return a added cube
  cube=createMyGroup(group)

  # add a IndexedFaceSet
  ifs=OpenMBV.ObjectFactory.create_IndexedFaceSet()
  ifs.setName("IFS")
  indices=[2, 6, 3, 1]
  print(indices)
  ifs.setIndices(indices)
  group.addObject(ifs)
   
  # create H5 and xml file
  group.setFileName("MBS_outfile.ombvx")
  group.write(True, True)

  # some action on the returned cube
  print(cube.getName())
  print(cube.getReferenceFrame())
  print(cube.getLength())
  cube.append([0.1, 1, 2, 3, 4, 5, 6, 0.25])


# a subroutine
def createMyGroup(g):
  # create a cube
  cube=OpenMBV.ObjectFactory.create_Cube()
  cube.setName("Box1")
  cube.setReferenceFrame(True)
  cube.setLength(1.234)
  
  # create a cuboid
  cuboid=OpenMBV.ObjectFactory.create_Cuboid()
  cuboid.setName("Box2")
  cuboid.setReferenceFrame(True)
  cuboid.setLength(1.234, 3, 4)
  
  # add cube and cuboid to group
  g.addObject(cube)
  g.addObject(cuboid)

  # some actions on cuboid
  print(cuboid.getLength())
  cuboid.setLength([5.6, 9.7, 3.5])
  print(cuboid.getLength())

  # return the cube
  return cube


main()
