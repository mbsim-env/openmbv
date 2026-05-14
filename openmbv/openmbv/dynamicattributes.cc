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
#include <openmbvcppinterface/dynamicattributes.h>

using namespace std;

namespace OpenMBVGUI {

DynamicAttributes::DynamicAttributes(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  da=static_pointer_cast<OpenMBV::DynamicAttributes>(obj);
}

DynamicAttributes::~DynamicAttributes() = default;

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
        auto o = getObjectByPath(p.path, this);
        if(o && !dynamic_cast<Obj>(o))
          msg(Error)<<"The object at path '"<<p.path<<"' is not of type Object."<<endl;
        pd.emplace_back(Data<Obj>{static_cast<Obj>(o), p.skip});
      }
    };
  
    convert(objectEnable, da->getObjectEnable());
    convert(bodyDrawMethod, da->getBodyDrawMethod());
    convert(dynamicColoredBodyTransparency, da->getDynamicColoredBodyTransparency());

    oldData.clear();
    oldData.resize(da->getDataSize(), numeric_limits<OpenMBV::Float>::quiet_NaN());
  }

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];
  auto data=da->getRow(frame);

  // set scene values
  size_t idxData = 0;
  auto upd = [this, &data, &idxData](auto &pd, const function<void(typename std::remove_reference_t<decltype(pd)>::value_type::type, double)>& set) {
    bool dataChanged = false;
    for(size_t idxPD=0; idxPD<pd.size(); ++idxPD) {
      if(!pd[idxPD].skip) {
        idxData++;
        dataChanged = oldData[idxData] != data[idxData];
        if(dataChanged)
          oldData[idxData] = data[idxData];
      }
      if(!dataChanged)
        continue;
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
