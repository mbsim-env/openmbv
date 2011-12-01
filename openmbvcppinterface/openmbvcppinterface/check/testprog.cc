#include "config.h"
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

  Group *g=new Group;
  g->setName("mygrp");
  g->setFileName("mygrp.ombv.xml");

    Cuboid *c2=new Cuboid;
    c2->setName("mycubeaa");
    c2->setLength(VectorParameter("length3",vector<double>(3,1.5)));
    g->addObject(c2);

    IvBody *iv=new IvBody;
    iv->setName("myiv");
    iv->setIvFileName("ivcube.iv");
    g->addObject(iv);

    Group *subg=new Group;
    subg->setName("mysubgrp");
    subg->setSeparateFile(true);
    g->addObject(subg);

      Cuboid *cX=new Cuboid;
      cX->setName("mycubeX");
      subg->addObject(cX);
      cX->setScaleFactor(ScalarParameter("scale",2));
      cX->setLocalFrame(true);

      Cuboid *c=new Cuboid;
      c->setName("mycubeaa");
      c->setHDF5LinkTarget(cX);
      subg->addObject(c);

      Cuboid *cZ=new Cuboid;
      cZ->setName("mycubeZ");
      cZ->setHDF5LinkTarget(cX);
      subg->addObject(cZ);

    Cuboid *c3=new Cuboid;
    c3->setName("mycube3");
    c3->setHDF5LinkTarget(cX);
    g->addObject(c3);

    Cube *cube=new Cube;
    cube->setName("mycube");
    cube->setLength(ScalarParameter("length",5));
    g->addObject(cube);

    Frame *frame=new Frame;
    frame->setName("myframe");
    frame->setSize(2);
    g->addObject(frame);

    Arrow *arrow=new Arrow;
    arrow->setName("myarrow");
    g->addObject(arrow);

    Frustum *cylinder=new Frustum;
    cylinder->setName("mycylinder");
    g->addObject(cylinder);
    
    Sphere *sphere=new Sphere;
    sphere->setName("mysphere");
    g->addObject(sphere);
    sphere->setMaximalColorValue(ScalarParameter("maxcolor",8));
    sphere->setMinimalColorValue(ScalarParameter("mincolor",5));
    
    Extrusion *extrusion=new Extrusion;
    extrusion->setName("myextrusion");
    PolygonPoint *point11=new PolygonPoint(0.6,0.2,0);
    PolygonPoint *point12=new PolygonPoint(0.6,0.2,0);
    PolygonPoint *point13=new PolygonPoint(0.6,0.2,0);
    vector<PolygonPoint*> *contour1=new vector<PolygonPoint*>;
    contour1->push_back(point11);
    contour1->push_back(point12);
    contour1->push_back(point13);
    extrusion->addContour(contour1);
    PolygonPoint *point21=new PolygonPoint(0.6,0.2,0);
    PolygonPoint *point22=new PolygonPoint(0.6,0.2,0);
    PolygonPoint *point23=new PolygonPoint(0.6,0.2,0);
    vector<PolygonPoint*> *contour2=new vector<PolygonPoint*>;
    contour2->push_back(point21);
    contour2->push_back(point22);
    contour2->push_back(point23);
    extrusion->addContour(contour2);
    g->addObject(extrusion);

    Rotation *rotation=new Rotation;
    rotation->setName("myrotation");
    PolygonPoint *point31=new PolygonPoint(0.6,0.2,0);
    PolygonPoint *point32=new PolygonPoint(0.6,0.2,0);
    PolygonPoint *point33=new PolygonPoint(0.6,0.2,0);
    vector<PolygonPoint*> *contour3=new vector<PolygonPoint*>;
    contour3->push_back(point31);
    contour3->push_back(point32);
    contour3->push_back(point33);
    rotation->setContour(contour3);
    g->addObject(rotation);
    
    InvisibleBody *invisiblebody=new InvisibleBody;
    invisiblebody->setName("myinvisiblebody");
    g->addObject(invisiblebody);
    
    CoilSpring *coilspring=new CoilSpring;
    coilspring->setName("mycoilspring");
    g->addObject(coilspring);
    
    CompoundRigidBody *crg=new CompoundRigidBody;
    crg->setName("mycrg");
      Rotation *rotationc=new Rotation;
      rotationc->setName("myrotationc");
      PolygonPoint *point41=new PolygonPoint(0.6,0.2,0);
      PolygonPoint *point42=new PolygonPoint(0.6,0.2,0);
      PolygonPoint *point43=new PolygonPoint(0.6,0.2,0);
      vector<PolygonPoint*> *contour4=new vector<PolygonPoint*>;
      contour4->push_back(point41);
      contour4->push_back(point42);
      contour4->push_back(point43);
      rotationc->setContour(contour4);
    crg->addRigidBody(rotationc);
      IvBody *ivc=new IvBody;
      ivc->setName("myivc");
      ivc->setIvFileName("ivcube.iv");
    crg->addRigidBody(ivc);
    g->addObject(crg);


  g->write();

  vector<double> row(8);
  for(int i=0; i<10; i++) {
    row[1]=i/10.0;
    c2->append(row);
    iv->append(row);
    cX->append(row);
    cube->append(row);
    frame->append(row);
    arrow->append(row);
    cylinder->append(row);
    sphere->append(row);
    extrusion->append(row);
    rotation->append(row);
    invisiblebody->append(row);
    coilspring->append(row);
    crg->append(row);
  }

  g->destroy();
  }
  cout<<"WALKHIERARCHY"<<endl;
  {

  Group *g=new Group;
  g->setFileName("mygrp.ombv.xml");
  g->read();
  walkHierarchy(g);
  
  g->destroy();

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
