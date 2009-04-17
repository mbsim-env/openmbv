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
#include "mainwindow.h"
#include <string>

using namespace std;

Object *ObjectFactory(TiXmlElement *element, H5::Group *h5Parent) {
  if(element->ValueStr()==MBVISNS"Group")
    return new Group(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Cuboid")
    return new Cuboid(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Cube")
    return new Cube(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Sphere")
    return new Sphere(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"InvisibleBody")
    return new InvisibleBody(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Frustum")
    return new Frustum(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"IvBody")
    return new IvBody(element, h5Parent);
  else if(element->ValueStr()==MBVISNS"Frame")
    return new Frame(element, h5Parent);
  MainWindow::getInstance()->getStatusBar()->showMessage(QString("ERROR: Unknown element: %1").arg(element->Value()), 2000);
  cout<<"ERROR: Unknown element: "<<element->ValueStr()<<endl;
  return 0;
}
