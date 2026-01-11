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
#include <boost/container_hash/hash.hpp>
#include "mainwindow.h"
#include "openmbvcppinterface/dynamicivbody.h"
#include <GL/glext.h>

using namespace std;

namespace OpenMBVGUI {

DynamicIvBody::DynamicIvBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  divb=std::static_pointer_cast<OpenMBV::DynamicIvBody>(obj);
  iconFile="dynamicivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  string fileName=divb->getIvFileName();

  if( divb->getStateOffSet().size() > 0 ) {
    data = std::vector<OpenMBV::Float>(divb->getStateOffSet().size()+1);

    for( size_t i = 0; i < divb->getStateOffSet().size(); ++i )
      data[i+1] = divb->getStateOffSet()[i];
    data[0] = 0;
  } else
    //h5 dataset
    data = divb->getRow(0);

  int rows=divb->getRows();
  double dt;
  if(rows>=2) dt=divb->getRow(1)[0]-divb->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineStyle->setName("openmbv_body_outline_style");
  soOutLineSwitch->setName("openmbv_body_outline_switch");

  if(divb->getScalarData()) {
    dataNodeScalar.resize(divb->getDataSize());
    for(size_t i=0; i<divb->getDataSize(); ++i) {
      dataNodeScalar[i] = new SoShaderParameter1f;
      soSep->addChild(dataNodeScalar[i]);
      dataNodeScalar[i]->setName(("openmbv_dynamicivbody_data_"+to_string(i)).c_str());
      dataNodeScalar[i]->value.setValue(data[i]);
    }
  }
  else {
    dataNodeVector = new SoShaderParameterArray1f;
    soSep->addChild(dataNodeVector);
    dataNodeVector->setName("openmbv_dynamicivbody_data");
    dataNodeVector->value.setNum(divb->getDataSize());
    dataNodeVector->value.setValuesPointer(divb->getDataSize(), data.data());
  }

  auto inFunc = [this](SoInput& in) {
    in.addReference("openmbv_body_outline_style", soOutLineStyle);
    in.addReference("openmbv_body_outline_switch", soOutLineSwitch);
    if(divb->getScalarData())
      for(size_t i=0; i<divb->getDataSize(); ++i)
        in.addReference(("openmbv_dynamicivbody_data_"+to_string(i)).c_str(), dataNodeScalar[i]);
    else
      in.addReference("openmbv_dynamicivbody_data", dataNodeVector);
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
    if(glGetError() != GL_NO_ERROR || maxFrag==-1 || maxVert==-1)
      throw runtime_error("Calling glGetIntegerv failed");
    size_t max = std::max(maxFrag, maxVert);
    if(( divb->getScalarData() && MainWindow::coinConsumedUniformBasicMachineUnits + divb->getDataSize()*4 > max) ||
       (!divb->getScalarData() && MainWindow::coinConsumedUniformBasicMachineUnits + ((divb->getDataSize()+3)/4)*4 > max))
      throw runtime_error("The number of dataSize of this DynamicIvBody is too large for the 'uniform' limit of your GPU.\n"
                          "(dataSize="+to_string((divb->getDataSize()))+"; limit="+
                                       to_string(max-MainWindow::coinConsumedUniformBasicMachineUnits)+")\n"
                          "(Switching from scalarData=true to scalarData=false will reduce the number by factor 4)\n");
  }

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];
  data=divb->getRow(frame);
  
  // set scene values
  if(divb->getScalarData())
    for(size_t i=0; i<divb->getDataSize(); ++i)
      dataNodeScalar[i]->value.setValue(data[i]);
  else
    dataNodeVector->value.setValuesPointer(divb->getDataSize(), data.data());

  return data[0];
}

}
