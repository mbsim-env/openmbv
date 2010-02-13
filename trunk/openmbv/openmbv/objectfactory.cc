/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "config.h"
#include "objectfactory.h"
#include "arrow.h"
#include "coilspring.h"
#include "compoundrigidbody.h"
#include "cube.h"
#include "cuboid.h"
#include "extrusion.h"
#include "frame.h"
#include "grid.h"
#include "frustum.h"
#include "group.h"
#include "invisiblebody.h"
#include "iostream"
#include "ivbody.h"
#include "mainwindow.h"
#include "nurbsdisk.h"
#include "path.h"
#include "sphere.h"
#include "spineextrusion.h"
#include <string>

using namespace std;

Object *ObjectFactory(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) {
  if(element->ValueStr()==OPENMBVNS"Group")
    return new Group(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Arrow")
    return new Arrow(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"CoilSpring")
    return new CoilSpring(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"CompoundRigidBody")
    return new CompoundRigidBody(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Cube")
    return new Cube(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Cuboid")
    return new Cuboid(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Extrusion")
    return new Extrusion(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Grid")
    return new Grid(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Frame")
    return new Frame(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Frustum")
    return new Frustum(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"IvBody")
    return new IvBody(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"InvisibleBody")
    return new InvisibleBody(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"NurbsDisk")
    return new NurbsDisk(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Path")
    return new Path(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"Sphere")
    return new Sphere(element, h5Parent, parentItem, soParent);
  else if(element->ValueStr()==OPENMBVNS"SpineExtrusion")
    return new SpineExtrusion(element, h5Parent, parentItem, soParent);
  QString str("ERROR: Unknown element: %1"); str=str.arg(element->Value());
  MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
  cout<<str.toStdString()<<endl;
  return 0;
}

