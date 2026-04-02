/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include "config.h"
#include "dynamiccoloredbody.h"
#include "mainwindow.h"
#include "dynamicattributes.h"
#include <boost/algorithm/string/split.hpp>
#include <openmbvcppinterface/dynamicattributes.h>

using namespace std;

namespace OpenMBVGUI {

DynamicAttributes::DynamicAttributes(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  da=static_pointer_cast<OpenMBV::DynamicAttributes>(obj);

  for(auto &p : da->getObjectEnable()) {
    auto o = getByPath(p);
    if(o && !dynamic_cast<Object*>(o))
      msg(Error)<<"The object at path '"<<p<<"' is not of type Object."<<endl;
    objectEnable.emplace_back(o);
  }
  for(auto &p : da->getBodyDrawMethod()) {
    auto o = getByPath(p);
    if(o && !dynamic_cast<Body*>(o))
      msg(Error)<<"The object at path '"<<p<<"' is not of type Body."<<endl;
    bodyDrawMethod.emplace_back(o);
  }
  for(auto &p : da->getDynamicColoredBodyTransparency()) {
    auto o = getByPath(p);
    if(o && !dynamic_cast<DynamicColoredBody*>(o))
      msg(Error)<<"The object at path '"<<p<<"' is not of type DynamicColoredBody."<<endl;
    dynamicColoredBodyTransparency.emplace_back(o);
  }
}

DynamicAttributes::~DynamicAttributes() = default;

Object* DynamicAttributes::getByPath(const std::string &path) {
  if(path[0]!='/' && path.substr(0,3)!="../") {
    msg(Error)<<"Illegal path '"<<path<<"'. Must start with '/' or '../'."<<endl;
    return nullptr;
  }

  // split path
  vector<string> pathVec;
  boost::split(pathVec, path, boost::is_any_of("/"));

  size_t iStart;
  Object* obj;
  if(path[0]=='/') {
    // search root
    obj = this;
    while(obj->QTreeWidgetItem::parent())
      obj = static_cast<Object*>(obj->QTreeWidgetItem::parent());
    iStart = 2;

    if(pathVec[1]!=obj->text(0).toStdString()) {
      msg(Error)<<"Illegal path '"<<path<<"'. Not found."<<endl;
      return nullptr;
    }
  }
  else {
    // search base
    obj = this;
    for(iStart=0; iStart<pathVec.size(); ++iStart)
      if(pathVec[iStart]=="..")
        obj = static_cast<Object*>(obj->QTreeWidgetItem::parent());
      else
        break;
  }

  // search path

  for(size_t i=iStart; i<pathVec.size(); ++i) {
    bool found=false;
    for(int c=0; c<obj->childCount(); ++c) {
      if(obj->child(c)->text(0).toStdString()==pathVec[i]) {
        obj = static_cast<Object*>(obj->child(c));
        found=true;
        break;
      }
    }
    if(!found) {
      msg(Error)<<"Illegal path '"<<path<<"'. Not found."<<endl;
      return nullptr;
    }
  }

  return obj;
}

double DynamicAttributes::update() {
  if(da->getRows()==0) return 0; // do nothing for environement objects

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];
  auto data=da->getRow(frame);

  // set scene values
  for(size_t i=0; i<objectEnable.size(); ++i)
    if(objectEnable[i])
      objectEnable[i]->soSwitch->whichChild.setValue(round(data[1+i])==0 ? SO_SWITCH_NONE : SO_SWITCH_ALL);
  for(size_t i=0; i<bodyDrawMethod.size(); ++i)
    if(bodyDrawMethod[i])
      static_cast<Body*>(bodyDrawMethod[i])->drawStyle->style.setValue(round(data[1+i]));
  for(size_t i=0; i<dynamicColoredBodyTransparency.size(); ++i)
    if(dynamicColoredBodyTransparency[i])
      static_cast<DynamicColoredBody*>(dynamicColoredBodyTransparency[i])->mat->transparency.setValue(data[1+i]);

  return data[0];
}

}
