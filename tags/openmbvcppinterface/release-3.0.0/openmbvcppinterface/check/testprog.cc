#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/cuboid.h>
#include <openmbvcppinterface/cube.h>
#include <openmbvcppinterface/frame.h>
#include <openmbvcppinterface/arrow.h>
#include <openmbvcppinterface/frustum.h>
#include <openmbvcppinterface/sphere.h>
#include <openmbvcppinterface/extrusion.h>
#include <openmbvcppinterface/rotation.h>
#include <openmbvcppinterface/invisiblebody.h>
#include <openmbvcppinterface/coilspring.h>
#include <openmbvcppinterface/objbody.h>
#include <openmbvcppinterface/compoundrigidbody.h>
#include <iostream>

using namespace OpenMBV;
using namespace std;

int main() {
  Group g;
  g.setName("mygrp");

    Cuboid c2;
    c2.setName("mycubeaa");
    g.addObject(&c2);

    Group subg;
    subg.setName("mysubgrp");
    subg.setSeparateFile(true);
    g.addObject(&subg);

      Cuboid cX;
      cX.setName("mycubeX");
      subg.addObject(&cX);

      Cuboid c;
      c.setName("mycubeaa");
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

    Cube cube;
    cube.setName("mycube");
    g.addObject(&cube);

    Frame frame;
    frame.setName("myframe");
    g.addObject(&frame);

    Arrow arrow;
    arrow.setName("myarrow");
    g.addObject(&arrow);

    Frustum cylinder;
    cylinder.setName("mycylinder");
    g.addObject(&cylinder);
    
    Sphere sphere;
    sphere.setName("mysphere");
    g.addObject(&sphere);
    
    Extrusion extrusion;
    extrusion.setName("myextrusion");
    PolygonPoint point(0.6,0.2,0);
    vector<PolygonPoint*> contour;
    contour.push_back(&point);
    contour.push_back(&point);
    contour.push_back(&point);
    extrusion.addContour(&contour);
    extrusion.addContour(&contour);
    g.addObject(&extrusion);

    Rotation rotation;
    rotation.setName("myrotation");
    rotation.setContour(&contour);
    g.addObject(&rotation);
    
    InvisibleBody invisiblebody;
    invisiblebody.setName("myinvisiblebody");
    g.addObject(&invisiblebody);
    
    CoilSpring coilspring;
    coilspring.setName("mycoilspring");
    g.addObject(&coilspring);
    
    ObjBody objobject;
    objobject.setName("myobjobject");
    g.addObject(&objobject);
    
    CompoundRigidBody crg;
    crg.setName("mycrg");
    crg.addRigidBody(&objobject);
    crg.addRigidBody(&rotation);
    g.addObject(&crg);


  g.initialize();

  vector<double> row(8);
  for(int i=0; i<10; i++) {
    c2.append(row);
    cX.append(row);
    cube.append(row);
    frame.append(row);
    arrow.append(row);
    cylinder.append(row);
    sphere.append(row);
    extrusion.append(row);
    rotation.append(row);
    invisiblebody.append(row);
    coilspring.append(row);
    objobject.append(row);
  }
}
