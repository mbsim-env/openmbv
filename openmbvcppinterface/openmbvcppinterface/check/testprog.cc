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
#include <openmbvcppinterface/compoundrigidbody.h>
#include <openmbvcppinterface/ivbody.h>
#include <iostream>

using namespace OpenMBV;
using namespace std;

void walkHierarchy(Group *grp);

int main() {

  cout<<"CREATE"<<endl;
  {

  Group g;
  g.setName("mygrp");

    Cuboid c2;
    c2.setName("mycubeaa");
    c2.setLength(VectorParameter("length3",vector<double>(3,1.5)));
    g.addObject(&c2);

    IvBody iv;
    iv.setName("myiv");
    iv.setIvFileName("ivcube.iv");
    g.addObject(&iv);

    Group subg;
    subg.setName("mysubgrp");
    subg.setSeparateFile(true);
    g.addObject(&subg);

      Cuboid cX;
      cX.setName("mycubeX");
      subg.addObject(&cX);
      cX.setScaleFactor(ScalarParameter("scale",2));
      cX.setLocalFrame(true);

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
    cube.setLength(ScalarParameter("length",5));
    g.addObject(&cube);

    Frame frame;
    frame.setName("myframe");
    frame.setSize(2);
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
    
    CompoundRigidBody crg;
    crg.setName("mycrg");
      Rotation rotationc;
      rotationc.setName("myrotationc");
      rotationc.setContour(&contour);
    crg.addRigidBody(&rotationc);
      IvBody ivc;
      ivc.setName("myivc");
      ivc.setIvFileName("ivcube.iv");
    crg.addRigidBody(&ivc);
    g.addObject(&crg);


  g.write();

  vector<double> row(8);
  for(int i=0; i<10; i++) {
    row[1]=i/10.0;
    c2.append(row);
    iv.append(row);
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
    crg.append(row);
  }

  }
  cout<<"WALKHIERARCHY"<<endl;
  {

  walkHierarchy(Group::read("mygrp.ombv.xml"));

  }
}

void walkHierarchy(Group *grp) {
  cout<<grp->getFullName()<<endl;
  vector<Object*> obj=grp->getObjects();
  for(size_t i=0; i<obj.size(); i++) {
    if(obj[i]->getClassName()=="Group")
      walkHierarchy((Group*)obj[i]);
    else {
      cout<<obj[i]->getFullName()<<" [rows="<<((Body*)obj[i])->getRows()<<"]"<<endl;
    }
  }
}
