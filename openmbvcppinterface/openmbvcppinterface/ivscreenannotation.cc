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
#include "group.h"
#include "ivscreenannotation.h"

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(IvScreenAnnotation, OPENMBV%"IvScreenAnnotation")

IvScreenAnnotation::IvScreenAnnotation() {
  scale1To1Center.resize(2);
  scale1To1Center[0]=0;
  scale1To1Center[1]=0;
}

DOMElement* IvScreenAnnotation::writeXMLFile(DOMNode *parent) {
  auto *e=Body::writeXMLFile(parent);
  if(scale1To1)
    E(e)->addElementText(OPENMBV%"scale1To1At", scale1To1Center);
  if(!ivFileName.empty())
    E(e)->addElementText(OPENMBV%"ivFileName", "'"+ivFileName+"'");
  else
    E(e)->addElementText(OPENMBV%"ivContent", "'"+ivContent+"'");

  // column labels
  for(auto &cl : columnLabels)
    E(e)->addElementText(OPENMBV%"columnLabel", "'"+cl+"'");
  for(auto &cl : columnIntLabels)
    E(e)->addElementText(OPENMBV%"columnIntLabel", "'"+cl+"'");
  for(auto &cl : columnStrLabels)
    E(e)->addElementText(OPENMBV%"columnStrLabel", "'"+cl+"'");

  return nullptr;
}

void IvScreenAnnotation::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);
  auto *e=E(element)->getFirstElementChildNamed(OPENMBV%"scale1To1At");
  if(!e) {
    setScale1To1(false);
  }
  else {
    setScale1To1(true);
    setScale1To1At(E(e)->getText<std::vector<double>>(2));
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ivFileName");
  if(e) {
    string str = X()%E(e)->getFirstTextChild()->getData();
    setIvFileName(E(e)->convertPath(str.substr(1,str.length()-2)).string());
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ivContent");
  if(e) {
    string str = X()%E(e)->getFirstTextChild()->getData();
    setIvContent(str.substr(1,str.length()-2));
  }

  // column labels
  vector<string> columnLabels;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"columnLabel");
  while(e) {
    auto str = E(e)->getText<string>();
    columnLabels.emplace_back(str.substr(1,str.length()-2));
    e = E(e)->getNextElementSiblingNamed(OPENMBV%"columnLabel");
  }
  setColumnLabels(columnLabels);

  vector<string> columnIntLabels;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"columnIntLabel");
  while(e) {
    auto str = E(e)->getText<string>();
    columnIntLabels.emplace_back(str.substr(1,str.length()-2));
    e = E(e)->getNextElementSiblingNamed(OPENMBV%"columnIntLabel");
  }
  setColumnIntLabels(columnIntLabels);

  vector<string> columnStrLabels;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"columnStrLabel");
  while(e) {
    auto str = E(e)->getText<string>();
    columnStrLabels.emplace_back(str.substr(1,str.length()-2));
    e = E(e)->getNextElementSiblingNamed(OPENMBV%"columnStrLabel");
  }
  setColumnStrLabels(columnStrLabels);
}

void IvScreenAnnotation::setScale1To1(bool scale1To1_) {
  scale1To1 = scale1To1_;
}

bool IvScreenAnnotation::getScale1To1() {
  return scale1To1;
}

void IvScreenAnnotation::setScale1To1At(const vector<double> &scale1To1Center_) {
  scale1To1 = true;
  scale1To1Center = scale1To1Center_;
}

vector<double> IvScreenAnnotation::getScale1To1At() {
  return scale1To1Center;
}

void IvScreenAnnotation::setColumnLabels(const std::vector<std::string> &columnLabels_) {
  columnLabels = columnLabels_;
  setEnvironment(columnLabels.size()==0 && columnIntLabels.size()==0 && columnStrLabels.size()==0);
}

void IvScreenAnnotation::addColumnLabel(const std::string &columnLabel) {
  columnLabels.emplace_back(columnLabel);
  setEnvironment(columnLabels.size()==0 && columnIntLabels.size()==0 && columnStrLabels.size()==0);
}

void IvScreenAnnotation::setColumnIntLabels(const std::vector<std::string> &columnIntLabels_) {
  columnIntLabels = columnIntLabels_;
  setEnvironment(columnLabels.size()==0 && columnIntLabels.size()==0 && columnStrLabels.size()==0);
}

void IvScreenAnnotation::addColumnIntLabel(const std::string &columnIntLabel) {
  columnIntLabels.emplace_back(columnIntLabel);
  setEnvironment(columnLabels.size()==0 && columnIntLabels.size()==0 && columnStrLabels.size()==0);
}

void IvScreenAnnotation::setColumnStrLabels(const std::vector<std::string> &columnStrLabels_) {
  columnStrLabels = columnStrLabels_;
  setEnvironment(columnLabels.size()==0 && columnIntLabels.size()==0 && columnStrLabels.size()==0);
}

void IvScreenAnnotation::addColumnStrLabel(const std::string &columnStrLabel) {
  columnStrLabels.emplace_back(columnStrLabel);
  setEnvironment(columnLabels.size()==0 && columnIntLabels.size()==0 && columnStrLabels.size()==0);
}

const std::vector<std::string>& IvScreenAnnotation::getColumnLabels() const {
  return columnLabels;
}

const std::vector<std::string>& IvScreenAnnotation::getColumnIntLabels() const {
  return columnIntLabels;
}

const std::vector<std::string>& IvScreenAnnotation::getColumnStrLabels() const {
  return columnStrLabels;
}

void IvScreenAnnotation::createHDF5File() {
  Body::createHDF5File();

  data=hdf5Group->createChildObject<H5::VectorSerie<Float> >("data")(1+columnLabels.size());
  vector<string> colNames(1+columnLabels.size());
  colNames[0] = "time";
  for(size_t i=0; i<columnLabels.size(); ++i)
    colNames[i+1] = columnLabels[i];
  data->setColumnLabel(colNames);

  try {
    dataInt=hdf5Group->createChildObject<H5::VectorSerie<int> >("dataInt")(columnIntLabels.size());
    vector<string> colIntNames(columnIntLabels.size());
    for(size_t i=0; i<columnIntLabels.size(); ++i)
      colIntNames[i] = columnIntLabels[i];
    dataInt->setColumnLabel(colIntNames);
  }
  catch(...) {
    dataInt = nullptr;
  }

  try {
    dataStr=hdf5Group->createChildObject<H5::VectorSerie<string> >("dataStr")(columnStrLabels.size());
    vector<string> colStrNames(columnStrLabels.size());
    for(size_t i=0; i<columnStrLabels.size(); ++i)
      colStrNames[i] = columnStrLabels[i];
    dataStr->setColumnLabel(colStrNames);
  }
  catch(...) {
    dataStr = nullptr;
  }
}

void IvScreenAnnotation::openHDF5File() {
  Body::openHDF5File();

  if(!hdf5Group) return;

  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<Float> >("data");

    // handle legacy data (no "time" columns as first col)
    if(columnLabels.size() == data->getColumns())
      msg(Deprecated) << "The HDF5 file does not contain a 'time' column as first columns: " << getFullName() << endl;
  }
  catch(...) {
    data=nullptr;
    msg(Debug)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }

  try {
    dataInt=hdf5Group->openChildObject<H5::VectorSerie<int> >("dataInt");
  }
  catch(...) {
    dataInt=nullptr;
  }

  try {
    dataStr=hdf5Group->openChildObject<H5::VectorSerie<string> >("dataStr");
  }
  catch(...) {
    dataStr=nullptr;
  }
}

}
