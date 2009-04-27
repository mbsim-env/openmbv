#include "config.h"
#include "objectfactory.h"
#include "iostream"
#include "group.h"
#include "cuboid.h"
#include "cube.h"
#include "sphere.h"
#include "invisiblebody.h"
#include "frustum.h"
#include "ivbody.h"
#include "frame.h"
#include "path.h"
#include "arrow.h"
#include "objbody.h"
#include "extrusion.h"
#include "compoundrigidbody.h"
#include "mainwindow.h"
#include <string>

using namespace std;

Object *ObjectFactory(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) {
  if(element->ValueStr()==OPENMBVNS"Group")
    return new Group(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Cuboid")
    return new Cuboid(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Cube")
    return new Cube(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Sphere")
    return new Sphere(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"InvisibleBody")
    return new InvisibleBody(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Frustum")
    return new Frustum(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"IvBody")
    return new IvBody(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Frame")
    return new Frame(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Path")
    return new Path(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Arrow")
    return new Arrow(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"ObjBody")
    return new ObjBody(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Extrusion")
    return new Extrusion(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"CompoundRigidBody")
    return new CompoundRigidBody(element, h5Parent, parentItem, soParent);
  MainWindow::getInstance()->getStatusBar()->showMessage(QString("ERROR: Unknown element: %1").arg(element->Value()), 2000);
  cout<<"ERROR: Unknown element: "<<element->ValueStr()<<endl;
  return 0;
}
