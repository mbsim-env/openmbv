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
  setEnvironment(columnLabels.size()==0);
}

const std::vector<std::string>& IvScreenAnnotation::getColumnLabels() const {
  return columnLabels;
}

void IvScreenAnnotation::createHDF5File() {
  Body::createHDF5File();

  data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+columnLabels.size());
  vector<string> colNames(1+columnLabels.size());
  colNames[0] = "time";
  for(size_t i=0; i<columnLabels.size(); ++i)
    colNames[i+1] = columnLabels[i];
  data->setColumnLabel(colNames);
}

void IvScreenAnnotation::openHDF5File() {
  Body::openHDF5File();

  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");

    // handle legacy data (no "time" columns as first col)
    if(columnLabels.size() == data->getColumns())
      msg(Deprecated) << "The HDF5 file does not contain a 'time' column as first columns: " << getFullName() << endl;
  }
  catch(...) {
    data=nullptr;
    msg(Debug)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }
}

}
