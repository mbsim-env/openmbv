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
}

DynamicAttributes::~DynamicAttributes() = default;

Object* DynamicAttributes::getByPath(const std::string &path) {
  if(path[0]!='/' && path.substr(0,3)!="../") {
    msg(Error)<<"Illegal path '"<<path<<"': must start with '/' or '../'."<<endl;
    return nullptr;
  }

  // split path
  vector<string> pathVec;
  boost::split(pathVec, path, boost::is_any_of("/"));

  size_t iStart = 0;
  Object* obj = nullptr;
  if(path[0]=='/' && path[1]=='/') {
    // search root
    obj = this;
    while(obj->QTreeWidgetItem::parent())
      obj = static_cast<Object*>(obj->QTreeWidgetItem::parent());
    iStart = 2;
  }
  else if(path[0]=='.' && path[1]=='.' && path[2]=='/') {
    // search base
    obj = this;
    for(iStart=0; iStart<pathVec.size(); ++iStart)
      if(pathVec[iStart]=="..")
        obj = static_cast<Object*>(obj->QTreeWidgetItem::parent());
      else
        break;
  }
  else {
    msg(Error)<<"Illegal path '"<<path<<"': must start with '../' or '//'."<<endl;
    return nullptr;
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
      msg(Error)<<"Illegal path '"<<path<<"': not found."<<endl;
      return nullptr;
    }
  }

  return obj;
}

double DynamicAttributes::update() {
  if(da->getRows()==0) return 0; // do nothing for environement objects

  if(!pathSearchDone) {
    pathSearchDone = true;

    auto convert = [this](auto &pd, const auto &dapd) {
      pd.clear();
      if(!dapd.empty() && dapd[0].skip)
        throw runtime_error("Illegal input: the first entry of a DynamicAttributes list cannot have skip=true.");
  
      using Obj = typename std::remove_reference_t<decltype(pd)>::value_type::type;
      for(auto &p : dapd) {
        auto o = getByPath(p.path);
        if(o && !dynamic_cast<Obj>(o))
          msg(Error)<<"The object at path '"<<p.path<<"' is not of type Object."<<endl;
        pd.emplace_back(Data<Obj>{static_cast<Obj>(o), p.skip});
      }
    };
  
    convert(objectEnable, da->getObjectEnable());
    convert(bodyDrawMethod, da->getBodyDrawMethod());
    convert(dynamicColoredBodyTransparency, da->getDynamicColoredBodyTransparency());
  }

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];
  auto data=da->getRow(frame);

  // set scene values
  size_t idxData = 0;
  auto upd = [&data, &idxData](auto &pd, const function<void(typename std::remove_reference_t<decltype(pd)>::value_type::type, double)>& set) {
    for(size_t idxPD=0; idxPD<pd.size(); ++idxPD) {
      if(!pd[idxPD].skip)
        idxData++;
      if(pd[idxPD].obj)
        set(pd[idxPD].obj, data[idxData]);
    }
  };

  upd(objectEnable, [](Object* o, double d){ o->soSwitch->whichChild.setValue(round(d)==0 ? SO_SWITCH_NONE : SO_SWITCH_ALL); });
  upd(bodyDrawMethod, [](Body *o, double d){ o->drawStyle->style.setValue(round(d)); });
  upd(dynamicColoredBodyTransparency, [](DynamicColoredBody *o, double d){ o->mat->transparency.setValue(d); });

  return data[0];
}

}
