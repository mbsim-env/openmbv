#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  undef __STRICT_ANSI__ // to define _controlfp which is not part of ANSI and hence not defined in mingw
#  include <cfloat>
#  define __STRICT_ANSI__
#endif
#include "config.h"
#include <cassert>
#include <cfenv>
#define _USE_MATH_DEFINES
#include <cmath>
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
#include <openmbvcppinterface/dynamicivbody.h>
#include <openmbvcppinterface/spineextrusion.h>
#include <iostream>

using namespace OpenMBV;
using namespace std;

void walkHierarchy(const shared_ptr<Group> &grp);
void dynamicivbody();
void spineextrusion();

int main() {
#ifdef _WIN32
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
  _controlfp(~(_EM_ZERODIVIDE | _EM_INVALID | _EM_OVERFLOW), _MCW_EM);
#else
  assert(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)!=-1);
#endif
  setlocale(LC_ALL, "C");

  cout<<"CREATE"<<endl;
  {

  shared_ptr<Group> g=ObjectFactory::create<Group>();
  g->setName("mygrp");
  g->setFileName("mygrp.ombvx");

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
    g->addObject(subg);

      shared_ptr<Cuboid> cX=ObjectFactory::create<Cuboid>();
      cX->setName("mycubeX");
      subg->addObject(cX);
      cX->setScaleFactor(2);
      cX->setLocalFrame(true);

      shared_ptr<Cuboid> c=ObjectFactory::create<Cuboid>();
      c->setName("mycubeaa");
      subg->addObject(c);

      shared_ptr<Cuboid> cZ=ObjectFactory::create<Cuboid>();
      cZ->setName("mycubeZ");
      subg->addObject(cZ);

    shared_ptr<Cuboid> c3=ObjectFactory::create<Cuboid>();
    c3->setName("mycube3");
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

  vector<Float> row(8);
  for(int i=0; i<10; i++) {
    row[1]=i/10.0;
    c2->append(row);
    iv->append(row);
    cX->append(row);
    c->append(row);
    cZ->append(row);
    c3->append(row);
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
  g->setFileName("mygrp.ombvx");
  g->read();
  walkHierarchy(g);
  
  }



  cout<<"COPYCTOR"<<endl;
  {

  shared_ptr<RigidBody> r1=ObjectFactory::create<Cube>();
  r1->setName("r1");
  r1->setScaleFactor(0.11);
  static_pointer_cast<Cube>(r1)->setLength(22.2);

  shared_ptr<RigidBody> r2=ObjectFactory::create<RigidBody>(r1);
  shared_ptr<Cube> c=static_pointer_cast<Cube>(r2);
  cout<<"name "<<r2->getName()<<endl;
  cout<<"scaleFactor "<<r2->getScaleFactor()<<endl;
  cout<<"length "<<c->getLength()<<endl;
  if(r2->getName()!="r1") return  1;
  if(r2->getScaleFactor()!=0.11) return  1;
  if(c->getLength()!=22.2) return  1;
  
  }

  dynamicivbody();
  spineextrusion();
}

void walkHierarchy(const shared_ptr<Group> &grp) {
  cout<<grp->getFullName()<<endl;
  vector<shared_ptr<Object> > obj=grp->getObjects();
  for(auto & o : obj) {
    shared_ptr<Group> g=dynamic_pointer_cast<Group>(o);
    if(g)
      walkHierarchy(g);
    else {
      shared_ptr<Body> b=dynamic_pointer_cast<Body>(o);
      cout<<o->getFullName()<<" [rows="<<b->getRows()<<"]"<<endl;
    }
  }
}

void dynamicivbody() {
  shared_ptr<Group> g=ObjectFactory::create<Group>();
  g->setName("dynamicivbody");
  g->setFileName("dynamicivbody.ombvx");
    auto sp(ObjectFactory::create<DynamicIvBody>());
    g->addObject(sp);
    sp->setName("ivobject");
    sp->setIvFileName("dynamicivbody.iv");
    int Nsp=2000;
    int Nc=200;
    int Tt=1000;
    sp->setDataSize(1+6*Nsp);
    auto contour = make_shared<std::vector<std::shared_ptr<PolygonPoint>>>();
    double r=0.1;
    contour->emplace_back(PolygonPoint::create(0,r,1));
    contour->emplace_back(PolygonPoint::create(0,0,1));
    contour->emplace_back(PolygonPoint::create(r,0,1));
    double da=M_PI/2/(Nc-2);
    for(double a=da; a<M_PI/2-da/2; a+=da)
      contour->emplace_back(PolygonPoint::create(r*cos(a),r*sin(a),0));
    std::reverse(contour->begin(), contour->end());
  g->write();

  vector<Float> data(1+6*Nsp);
  double Tend=1;
  for(double t=0; t<Tend; t+=Tend/Tt) {
    data[0]=t;
    for(int Isp=0; Isp<Nsp; ++Isp) {
      double R=0.3*cos(2*M_PI*10*t);
      double x=static_cast<double>(Isp)/Nsp*2*M_PI;
      data[6*Isp+1] = x;
      data[6*Isp+2] = R*sin(x);
      data[6*Isp+3] = 0;
      data[6*Isp+4] = 0;
      data[6*Isp+5] = 0;
      data[6*Isp+6] = M_PI/2+atan(R*cos(x));
    }
    sp->append(data);
  }
}

void spineextrusion() {
  shared_ptr<Group> g=ObjectFactory::create<Group>();
  g->setName("spineextrusion");
  g->setFileName("spineextrusion.ombvx");
    auto sp(ObjectFactory::create<SpineExtrusion>());
    g->addObject(sp);
    sp->setName("ivobject");
    int Nsp=2000;
    int Nc=200;
    int Tt=1000;
    auto contour = make_shared<std::vector<std::shared_ptr<PolygonPoint>>>();
    double r=0.1;
    contour->emplace_back(PolygonPoint::create(0,r,1));
    contour->emplace_back(PolygonPoint::create(0,0,1));
    contour->emplace_back(PolygonPoint::create(r,0,1));
    double da=M_PI/2/(Nc-2);
    for(double a=da; a<M_PI/2-da/2; a+=da)
      contour->emplace_back(PolygonPoint::create(r*cos(a),r*sin(a),0));
    std::reverse(contour->begin(), contour->end());
    sp->setNumberOfSpinePoints(Nsp);
    sp->setDiffuseColor(120.0/360,1,1);
    sp->setContour(contour);
    sp->setCrossSectionOrientation(SpineExtrusion::cardanWrtWorldShader);
    sp->setCounterClockWise(true);

  g->write();

  vector<Float> data(1+6*Nsp);
  double Tend=1;
  for(double t=0; t<Tend; t+=Tend/Tt) {
    data[0]=t;
    for(int Isp=0; Isp<Nsp; ++Isp) {
      double R=0.3*cos(2*M_PI*10*t);
      double x=static_cast<double>(Isp)/Nsp*2*M_PI;
      data[6*Isp+1] = x;
      data[6*Isp+2] = R*sin(x);
      data[6*Isp+3] = 0;
      data[6*Isp+4] = 0;
      data[6*Isp+5] = 0;
      data[6*Isp+6] = M_PI/2+atan(R*cos(x));
    }
    sp->append(data);
  }
}
