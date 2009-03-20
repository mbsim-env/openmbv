#include <amviscppinterface/group.h>
#include <amviscppinterface/cuboid.h>
#include <iostream>

using namespace AMVis;
using namespace std;

int main() {
  Group g;
  g.setName("mygrp");

    Cuboid c2;
    c2.setName("mycube");
    g.addObject(&c2);

    Group subg;
    subg.setName("mysubgrp");
    subg.setSeparateFile(true);
    g.addObject(&subg);

      Cuboid cX;
      cX.setName("mycubeX");
      subg.addObject(&cX);

      Cuboid c;
      c.setName("mycube");
      c.setHDF5LinkTarget(&cX);
      subg.addObject(&c);

      Cuboid cZ;
      cZ.setName("mycubeZ");
      cZ.setHDF5LinkTarget(&cX);
      subg.addObject(&cZ);

    Cuboid c3;
    c3.setName("mycube3");
    c3.setHDF5LinkTarget(&cX);
    g.addObject(&c3);

    
  g.initialize();

  vector<double> row(8);
  for(int i=0; i<10; i++) {
    c2.append(row);
    cX.append(row);
  }
}
