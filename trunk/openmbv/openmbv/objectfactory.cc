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
#include "rotation.h"
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

Object *ObjectFactory(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent) {
  if(obj->getClassName()=="Group")
    return new Group(obj, parentItem, soParent);
  else if(obj->getClassName()=="Arrow")
    return new Arrow(obj, parentItem, soParent);
  else if(obj->getClassName()=="CoilSpring")
    return new CoilSpring(obj, parentItem, soParent);
  else if(obj->getClassName()=="CompoundRigidBody")
    return new CompoundRigidBody(obj, parentItem, soParent);
  else if(obj->getClassName()=="Cube")
    return new Cube(obj, parentItem, soParent);
  else if(obj->getClassName()=="Cuboid")
    return new Cuboid(obj, parentItem, soParent);
  else if(obj->getClassName()=="Extrusion")
    return new Extrusion(obj, parentItem, soParent);
  else if(obj->getClassName()=="Rotation")
    return new Rotation(obj, parentItem, soParent);
  else if(obj->getClassName()=="Grid")
    return new Grid(obj, parentItem, soParent);
  else if(obj->getClassName()=="Frame")
    return new Frame(obj, parentItem, soParent);
  else if(obj->getClassName()=="Frustum")
    return new Frustum(obj, parentItem, soParent);
  else if(obj->getClassName()=="IvBody")
    return new IvBody(obj, parentItem, soParent);
  else if(obj->getClassName()=="InvisibleBody")
    return new InvisibleBody(obj, parentItem, soParent);
  else if(obj->getClassName()=="NurbsDisk")
    return new NurbsDisk(obj, parentItem, soParent);
  else if(obj->getClassName()=="Path")
    return new Path(obj, parentItem, soParent);
  else if(obj->getClassName()=="Sphere")
    return new Sphere(obj, parentItem, soParent);
  else if(obj->getClassName()=="SpineExtrusion")
    return new SpineExtrusion(obj, parentItem, soParent);
  QString str("ERROR: Unknown OpenMBV::Object: %1"); str=str.arg(obj->getClassName().c_str());
  MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
  cout<<str.toStdString()<<endl;
  return 0;
}

