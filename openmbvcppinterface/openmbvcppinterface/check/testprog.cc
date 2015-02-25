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
using namespace boost;

void walkHierarchy(const shared_ptr<Group> &grp);

int main() {

  cout<<"CREATE"<<endl;
  {

  shared_ptr<Group> g=ObjectFactory::create<Group>();
  g->setName("mygrp");
  g->setFileName("mygrp.ombv.xml");

    shared_ptr<Cuboid> c2=ObjectFactory::create<Cuboid>();
    c2->setName("mycubeaa");
    c2->setLength(vector<double>(3,1.5));
    g->addObject(c2);

    shared_ptr<IvBody> iv=ObjectFactory::create<IvBody>();
    iv->setName("myiv");
    iv->setIvFileName("ivcube.iv");
    g->addObject(iv);

    shared_ptr<Group> subg=ObjectFactory::create<Group>();
    subg->setName("mysubgrp");
    subg->setSeparateFile(true);
    g->addObject(subg);

      shared_ptr<Cuboid> cX=ObjectFactory::create<Cuboid>();
      cX->setName("mycubeX");
      subg->addObject(cX);
      cX->setScaleFactor(2);
      cX->setLocalFrame(true);

      shared_ptr<Cuboid> c=ObjectFactory::create<Cuboid>();
      c->setName("mycubeaa");
      c->setHDF5LinkTarget(cX);
      subg->addObject(c);

      shared_ptr<Cuboid> cZ=ObjectFactory::create<Cuboid>();
      cZ->setName("mycubeZ");
      cZ->setHDF5LinkTarget(cX);
      subg->addObject(cZ);

    shared_ptr<Cuboid> c3=ObjectFactory::create<Cuboid>();
    c3->setName("mycube3");
    c3->setHDF5LinkTarget(cX);
    g->addObject(c3);

    shared_ptr<Cube> cube=ObjectFactory::create<Cube>();
    cube->setName("mycube");
    cube->setLength(5);
    g->addObject(cube);

    shared_ptr<Frame> frame=ObjectFactory::create<Frame>();
    frame->setName("myframe");
    frame->setSize(2);
    g->addObject(frame);

    shared_ptr<Arrow> arrow=ObjectFactory::create<Arrow>();
    arrow->setName("myarrow");
    g->addObject(arrow);

    shared_ptr<Frustum> cylinder=ObjectFactory::create<Frustum>();
    cylinder->setName("mycylinder");
    g->addObject(cylinder);
    
    shared_ptr<Sphere> sphere=ObjectFactory::create<Sphere>();
    sphere->setName("mysphere");
    g->addObject(sphere);
    sphere->setMaximalColorValue(8);
    sphere->setMinimalColorValue(5);
    
    shared_ptr<Extrusion> extrusion=ObjectFactory::create<Extrusion>();
    extrusion->setName("myextrusion");
    shared_ptr<PolygonPoint> point11=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<PolygonPoint> point12=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<PolygonPoint> point13=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<vector<shared_ptr<PolygonPoint> > > contour1=make_shared<vector<shared_ptr<PolygonPoint> > >();
    contour1->push_back(point11);
    contour1->push_back(point12);
    contour1->push_back(point13);
    extrusion->addContour(contour1);
    shared_ptr<PolygonPoint> point21=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<PolygonPoint> point22=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<PolygonPoint> point23=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<vector<shared_ptr<PolygonPoint> > > contour2=make_shared<vector<shared_ptr<PolygonPoint> > >();
    contour2->push_back(point21);
    contour2->push_back(point22);
    contour2->push_back(point23);
    extrusion->addContour(contour2);
    g->addObject(extrusion);

    shared_ptr<Rotation> rotation=ObjectFactory::create<Rotation>();
    rotation->setName("myrotation");
    shared_ptr<PolygonPoint> point31=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<PolygonPoint> point32=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<PolygonPoint> point33=PolygonPoint::create(0.6,0.2,0);
    shared_ptr<vector<shared_ptr<PolygonPoint> > > contour3=make_shared<vector<shared_ptr<PolygonPoint> > >();
    contour3->push_back(point31);
    contour3->push_back(point32);
    contour3->push_back(point33);
    rotation->setContour(contour3);
    g->addObject(rotation);
    
    shared_ptr<InvisibleBody> invisiblebody=ObjectFactory::create<InvisibleBody>();
    invisiblebody->setName("myinvisiblebody");
    g->addObject(invisiblebody);
    
    shared_ptr<CoilSpring> coilspring=ObjectFactory::create<CoilSpring>();
    coilspring->setName("mycoilspring");
    g->addObject(coilspring);
    
    shared_ptr<CompoundRigidBody> crb=ObjectFactory::create<CompoundRigidBody>();
    crb->setName("mycrb");
      shared_ptr<Rotation> rotationc=ObjectFactory::create<Rotation>();
      rotationc->setName("myrotationc");
      shared_ptr<PolygonPoint> point41=PolygonPoint::create(0.6,0.2,0);
      shared_ptr<PolygonPoint> point42=PolygonPoint::create(0.6,0.2,0);
      shared_ptr<PolygonPoint> point43=PolygonPoint::create(0.6,0.2,0);
      shared_ptr<vector<shared_ptr<PolygonPoint> > > contour4=make_shared<vector<shared_ptr<PolygonPoint> > >();
      contour4->push_back(point41);
      contour4->push_back(point42);
      contour4->push_back(point43);
      rotationc->setContour(contour4);
    crb->addRigidBody(rotationc);
      shared_ptr<IvBody> ivc=ObjectFactory::create<IvBody>();
      ivc->setName("myivc");
      ivc->setIvFileName("ivcube.iv");
    crb->addRigidBody(ivc);
    g->addObject(crb);


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
    crb->append(row);
  }

  }
  cout<<"WALKHIERARCHY"<<endl;
  {

  shared_ptr<Group> g=ObjectFactory::create<Group>();
  g->setFileName("mygrp.ombv.xml");
  g->read();
  walkHierarchy(g);
  
  }
}

void walkHierarchy(const shared_ptr<Group> &grp) {
  cout<<grp->getFullName()<<endl;
  vector<shared_ptr<Object> > obj=grp->getObjects();
  for(size_t i=0; i<obj.size(); i++) {
    shared_ptr<Group> g=dynamic_pointer_cast<Group>(obj[i]);
    if(g)
      walkHierarchy(g);
    else {
      shared_ptr<Body> b=dynamic_pointer_cast<Body>(obj[i]);
      cout<<obj[i]->getFullName()<<" [rows="<<b->getRows()<<"]"<<endl;
    }
  }
}
