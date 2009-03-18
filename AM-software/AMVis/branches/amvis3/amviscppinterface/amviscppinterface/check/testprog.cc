#include <amviscppinterface/group.h>
#include <amviscppinterface/cuboid.h>
#include <iostream>

using namespace AMVis;
using namespace std;

int main() {
  Group g("mygrp");

    Cuboid c2("mycube");
    g.addObject(&c2);

    Group subg("mysubgrp");
    subg.setSeparateFile(true);
    g.addObject(&subg);

      Cuboid cX("mycubeX");
      subg.addObject(&cX);

      Cuboid c("mycube");
      c.setHDF5Link(&cX);
      subg.addObject(&c);

      Cuboid cZ("mycubeZ");
      cZ.setHDF5Link(&cX);
      subg.addObject(&cZ);

    Cuboid c3("mycube3");
    c3.setHDF5Link(&cX);
    g.addObject(&c3);

    
  g.initialize();

  vector<double> row(8);
  for(int i=0; i<10; i++) {
    c2.append(row);
    cX.append(row);
  }
}
