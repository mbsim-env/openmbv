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
#include "dynamicivbody.h"

#include <Inventor/nodes/SoShaderParameter.h>
#include <Inventor/nodes/SoGeoOrigin.h>
#include <Inventor/nodes/SoInfo.h>
#include <boost/container_hash/hash.hpp>
#include "mainwindow.h"
#include "openmbvcppinterface/dynamicivbody.h"
#include <GL/glext.h>
#include <QMessageBox>

using namespace std;

namespace OpenMBVGUI {

DynamicIvBody::DynamicIvBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  divb=std::static_pointer_cast<OpenMBV::DynamicIvBody>(obj);
  iconFile="dynamicivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  string fileName=divb->getIvFileName();

  auto create = [](auto &data, const std::remove_reference_t<decltype(data)> &stateOffset,
                               const std::remove_reference_t<decltype(data)> &row0){
    if( stateOffset.size() > 0 ) {
      using Type = std::remove_reference_t<decltype(data)>;

      data = Type(stateOffset.size()+1);

      for( size_t i = 0; i < stateOffset.size(); ++i )
        data[i+1] = stateOffset[i];
      data[0] = typename Type::value_type();
    } else
      //h5 dataset
      data = row0;
  };
  create(data   , divb->getStateOffSet()   , divb->getRow(0));
  create(dataInt, divb->getStateIntOffSet(), divb->getRowInt(0));
  create(dataStr, divb->getStateStrOffSet(), divb->getRowStr(0));

  int rows=divb->getRows();
  double dt;
  if(rows>=2) dt=divb->getRow(1)[0]-divb->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineStyle->setName("openmbv_body_outline_style");
  soOutLineSwitch->setName("openmbv_body_outline_switch");

  auto swSep = new SoSeparator;
  soSep->addChild(swSep);
  auto sw = new SoSwitch;
  swSep->addChild(sw);
  sw->whichChild = SO_SWITCH_NONE;

  if(divb->getScalarData()) {
    auto scalar = [sw](auto &dataNodeScalar, size_t size, const auto &data, const string &name, const function<void(int)> &set) {
      dataNodeScalar.resize(size);
      for(size_t i=0; i<size; ++i) {
        dataNodeScalar[i] = new std::remove_pointer_t<typename std::remove_reference_t<decltype(dataNodeScalar)>::value_type>;
        sw->addChild(dataNodeScalar[i]);
        dataNodeScalar[i]->setName(("openmbv_dynamicivbody_"+name+"_"+to_string(i)).c_str());
        set(i);
      }
    };
    scalar(dataNodeScalar   , divb->getDataSize()   , data   , "data"   ,
           [this](int i){ dataNodeScalar[i]->value.setValue(data[i]); });
    scalar(dataIntNodeScalar, divb->getDataIntSize(), dataInt, "dataInt",
           [this](int i){ dataIntNodeScalar[i]->value.setValue(dataInt[i]); });
    scalar(dataStrNodeScalar, divb->getDataStrSize(), dataStr, "dataStr",
           [this](int i){ dataStrNodeScalar[i]->string.setValue(dataStr[i].c_str()); });
  }
  else {
    dataNodeVector = new SoShaderParameterArray1f;
    sw->addChild(dataNodeVector);
    dataNodeVector->setName("openmbv_dynamicivbody_data");
    dataNodeVector->value.setNum(divb->getDataSize());
    dataNodeVector->value.setValuesPointer(divb->getDataSize(), data.data());

    dataIntNodeVector = new SoShaderParameterArray1i;
    sw->addChild(dataIntNodeVector);
    dataIntNodeVector->setName("openmbv_dynamicivbody_dataInt");
    dataIntNodeVector->value.setNum(divb->getDataIntSize());
    dataIntNodeVector->value.setValuesPointer(divb->getDataIntSize(), dataInt.data());

    dataStrNodeVector = new SoAsciiText;
    sw->addChild(dataStrNodeVector);
    dataStrNodeVector->setName("openmbv_dynamicivbody_dataStr");
    dataStrNodeVector->string.setNum(divb->getDataStrSize());
    for(size_t i=0; i<divb->getDataStrSize(); ++i)
      dataStrNodeVector->string.set1Value(i, dataStr[i].c_str());
  }

  auto inFunc = [this](SoInput& in) {
    in.addReference("openmbv_body_outline_style", soOutLineStyle);
    in.addReference("openmbv_body_outline_switch", soOutLineSwitch);
    if(divb->getScalarData()) {
      for(size_t i=0; i<divb->getDataSize(); ++i)
        in.addReference(("openmbv_dynamicivbody_data_"+to_string(i)).c_str(), dataNodeScalar[i]);
      for(size_t i=0; i<divb->getDataIntSize(); ++i)
        in.addReference(("openmbv_dynamicivbody_dataInt_"+to_string(i)).c_str(), dataIntNodeScalar[i]);
      for(size_t i=0; i<divb->getDataStrSize(); ++i)
        in.addReference(("openmbv_dynamicivbody_dataStr_"+to_string(i)).c_str(), dataStrNodeScalar[i]);
    }
    else {
      in.addReference("openmbv_dynamicivbody_data"   , dataNodeVector);
      in.addReference("openmbv_dynamicivbody_dataInt", dataIntNodeVector);
      in.addReference("openmbv_dynamicivbody_dataStr", dataStrNodeVector);
    }
  };
  SoGroup *soIv;
  // load the IV content (without caching since element specific node references must be resolved)
  if(!fileName.empty())
    soIv=Utils::SoDBreadAllFileNameCached(fileName, {/*no cache*/}, inFunc);
  else
    soIv=Utils::SoDBreadAllContentCached(divb->getIvContent(), {/*no cache*/}, inFunc);
  if(!soIv)
    return;
  soSep->addChild(soIv);
}

DynamicIvBody::~DynamicIvBody() = default;

double DynamicIvBody::update() {
  if(divb->getRows()==0) return 0; // do nothing for environement objects

  // check OpenGL limits
  if(!runtimeCheckDone) {
    runtimeCheckDone = true;
    assert(glGetString(GL_VERSION));
    GLint maxFrag=-1, maxVert=-1;
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &maxFrag);
    glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &maxVert);
    if(glGetError() != GL_NO_ERROR || maxFrag==-1 || maxVert==-1) {
      string msg("Calling glGetIntegerv failed\nExiting now!");
      QMessageBox::critical(nullptr, "Critical Error", msg.c_str());
      throw runtime_error(msg);
    }
    size_t max = std::max(maxFrag, maxVert);
    if(( divb->getScalarData() && MainWindow::coinConsumedUniformBasicMachineUnits + divb->getDataSize()*4 > max) ||
       (!divb->getScalarData() && MainWindow::coinConsumedUniformBasicMachineUnits + ((divb->getDataSize()+3)/4)*4 > max)) {
      auto msg("The number of dataSize of this DynamicIvBody is too large for the 'uniform' limit of your GPU.\n"
               "(dataSize="+to_string((divb->getDataSize()))+"; limit="+
                            to_string(max-MainWindow::coinConsumedUniformBasicMachineUnits)+")\n"
               "(Switching from scalarData=true to scalarData=false will reduce the number by factor 4)\n"
               "Exiting now");
      QMessageBox::critical(nullptr, "Critical Error", msg.c_str());
      throw runtime_error(msg);
    }
  }

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];

  double ret = 0;

  {
    data=divb->getRow(frame);
    // set scene values
    if(divb->getScalarData())
      for(size_t i=0; i<divb->getDataSize(); ++i)
        dataNodeScalar[i]->value.setValue(data[i]);
    else
      dataNodeVector->value.setValuesPointer(divb->getDataSize(), data.data());
    ret = data[0];
  }
  if(divb->getDataIntSize()>0) {
    dataInt=divb->getRowInt(frame);
    // set scene values
    if(divb->getScalarData())
      for(size_t i=0; i<divb->getDataIntSize(); ++i)
        dataIntNodeScalar[i]->value.setValue(dataInt[i]);
    else
      dataIntNodeVector->value.setValuesPointer(divb->getDataIntSize(), dataInt.data());
  }
  if(divb->getDataStrSize()>0) {
    dataStr=divb->getRowStr(frame);
    // set scene values
    if(divb->getScalarData())
      for(size_t i=0; i<divb->getDataStrSize(); ++i)
        dataStrNodeScalar[i]->string.setValue(dataStr[i].c_str());
    else {
      for(size_t i=0; i<divb->getDataStrSize(); ++i)
        dataStrNodeVector->string.set1Value(i, dataStr[i].c_str());
    }
  }

  return ret;
}

}
