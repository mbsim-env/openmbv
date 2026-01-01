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
#include <Inventor/actions/SoSearchAction.h>

#include <Inventor/nodes/SoShaderParameter.h>
#include <boost/container_hash/hash.hpp>
#include "mainwindow.h"
#include "openmbvcppinterface/dynamicivbody.h"

using namespace std;

namespace OpenMBVGUI {

DynamicIvBody::DynamicIvBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  divb=std::static_pointer_cast<OpenMBV::DynamicIvBody>(obj);
  iconFile="dynamicivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  string fileName=divb->getIvFileName();

  if( divb->getStateOffSet().size() > 0 ) {
    data = std::vector<double>(divb->getStateOffSet().size()+1);

    for( size_t i = 0; i < divb->getStateOffSet().size(); ++i )
      data[i+1] = divb->getStateOffSet()[i];
    data[0] = 0;
  } else
    //h5 dataset
    data = divb->getRow(0);

  auto hashData = make_tuple(
    divb->getRemoveNodesByName(),
    divb->getRemoveNodesByType()
  );

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineColorMatGrp->setName("openmbv_body_outline_style");

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
    datamfmf.resize(divb->getDataSize());
    for(size_t i=0; i<divb->getDataSize(); ++i)
      datamfmf[i] = data[i];
    dataNodeVector->value.setValuesPointer(divb->getDataSize(), datamfmf.data());
  }

  auto inFunc = [this](SoInput& in) {
    in.addReference("openmbv_body_outline_style", soOutLineColorMatGrp);
    if(divb->getScalarData())
      for(size_t i=0; i<divb->getDataSize(); ++i)
        in.addReference(("openmbv_dynamicivbody_data_"+to_string(i)).c_str(), dataNodeScalar[i]);
    else
      in.addReference("openmbv_dynamicivbody_data", dataNodeVector);
  };
  SoGroup *soIv;
  if(!fileName.empty())
    soIv=Utils::SoDBreadAllFileNameCached(fileName, boost::hash<decltype(hashData)>{}(hashData), inFunc);
  else
    soIv=Utils::SoDBreadAllContentCached(divb->getIvContent(), boost::hash<decltype(hashData)>{}(hashData), inFunc);
  if(!soIv)
    return;
  soSep->addChild(soIv);

  // search and remove specific nodes
  auto removeNode=[soIv, &fileName, this](const function<void(SoSearchAction &sa)> &find){
    SoSearchAction sa;
    sa.setInterest(SoSearchAction::ALL);
    find(sa);
    sa.apply(soIv);
    auto pl = sa.getPaths();
    for(int i=0; i<pl.getLength(); ++i) {
      msg(Info)<<"Removing the following node for DynamicIvBody from file '"<<fileName<<"':"<<endl;
      auto *p = pl[i];
      for(int j=1; j<p->getLength(); ++j) {
        auto *n = p->getNode(j);
        msg(Info)<<string(2*j, ' ')<<"- Name: '"<<n->getName()<<"'; Type: '"<<n->getTypeId().getName().getString()<<"'"<<endl;
      }
      static_cast<SoGroup*>(p->getNodeFromTail(1))->removeChild(p->getIndexFromTail(0));
    }
  };
  // remove nodes by name
  for(auto &name : divb->getRemoveNodesByName())
    removeNode([&name](auto &sa){ sa.setName(name.c_str()); });
  // remove nodes by type
  for(auto type : divb->getRemoveNodesByType()) {
    // We fix the hacked SoVRMLBackground2 node which overrids Background
    if(type=="Background" || type=="VRMLBackground" || type=="SoVRMLBackground")
      type="SoVRMLBackground2";
    removeNode([&type](auto &sa){ sa.setType(SoType::fromName(type.c_str())); });
  }
}

DynamicIvBody::~DynamicIvBody() = default;

double DynamicIvBody::update() {
  if(divb->getRows()==0) return 0; // do nothing for environement objects

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];
  data=divb->getRow(frame);
  
  // set scene values
  if(divb->getScalarData())
    for(size_t i=0; i<divb->getDataSize(); ++i)
      dataNodeScalar[i]->value.setValue(data[i]);
  else {
    datamfmf.resize(divb->getDataSize());
    for(size_t i=0; i<divb->getDataSize(); ++i)
      datamfmf[i] = data[i];
    dataNodeVector->value.setValuesPointer(divb->getDataSize(), datamfmf.data());
  }

  return data[0];
}

}
