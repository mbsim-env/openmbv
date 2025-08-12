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
#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/body.h>
#include <openmbvcppinterface/objectfactory.h>
#include <hdf5serie/file.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace {
  boost::filesystem::path getPreSWMRFileName(const boost::filesystem::path &fileName) {
    return (fileName.parent_path()/(fileName.stem().string()+".preSWMR"+fileName.extension().string())).string();
  }
}

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Group, OPENMBV%"Group")

Group::Group() : expandStr("true") {
}

Group::~Group() {
  if(ombvxRenameNeeded)
    boost::filesystem::rename(getPreSWMRFileName(fileName), fileName);
}

void Group::addObject(const shared_ptr<Object>& newObject) {
  if(newObject->name.empty()) throw runtime_error("object to add must have a name");
  for(auto & i : object)
    if(i->name==newObject->name)
      throw runtime_error("A object of name "+i->name+" already exists.");
  object.push_back(newObject);
  newObject->parent=shared_from_this();
}

string Group::getFullName() {
  if(fullName.empty()) {
    std::shared_ptr<Group> p=parent.lock();
    if(p)
      fullName = p->getFullName()+"/"+name;
    else
      fullName = fileName.empty() ? name : fileName.string();
  }
  return fullName;
}

DOMElement *Group::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Object::writeXMLFile(parent);
  E(e)->setAttribute("expand", expandStr);
  for(auto & i : object)
    i->writeXMLFile(e);
  return nullptr;
}

void Group::createHDF5File() {
  std::shared_ptr<Group> p=parent.lock();
  hdf5Group=p->hdf5Group->createChildObject<H5::Group>(name)();
  for(auto & i : object)
    if(!i->getEnvironment()) i->createHDF5File();
}

void Group::openHDF5File() {
  hdf5Group=nullptr;
  std::shared_ptr<Group> p=parent.lock();
  if(!p)
    throw runtime_error("This Group is not a top level group, something is wrong.");
  else {
    try {
      if(!getEnvironment())
        hdf5Group=p->hdf5Group->openChildObject<H5::Group>(name);
    }
    catch(...) {
      msg(Debug)<<"Unable to open the HDF5 Group '"<<name<<"'. Using 0 for all data."<<endl;
    }
  }
  if(hdf5Group)
    for(auto & i : object)
      i->openHDF5File();
}

void Group::writeXML() {
  // write .ombvx file
  shared_ptr<DOMParser> parser=DOMParser::create();
  shared_ptr<DOMDocument> xmlFile=parser->createDocument();
  DOMElement *parent=Object::writeXMLFile(xmlFile.get());
  E(parent)->setAttribute("expand", expandStr);
  for(auto & i : object)
    i->writeXMLFile(parent);
  DOMParser::serialize(xmlFile.get(), getPreSWMRFileName(fileName).string());
  ombvxRenameNeeded = true;
}

void Group::initializeUsingXML(DOMElement *element) {
  Object::initializeUsingXML(element);
  if(E(element)->hasAttribute("expand") && 
     (E(element)->getAttribute("expand")=="false" || E(element)->getAttribute("expand")=="0"))
    setExpand(false);
  auto ofn = E(element)->getEmbedData("MBXMLUtils_OriginalFilename");
  if(!ofn.empty())
    fileName=ofn;

  DOMElement *e;
  e=element->getFirstElementChild();
  while (e) {
    if(E(e)->getTagName()==PV%"evaluator") { // skip the pv:evaluator element
      e=e->getNextElementSibling();
      continue;
    }
    shared_ptr<Object> obj=ObjectFactory::create<Object>(e);
    obj->initializeUsingXML(e);
    addObject(obj);
    e=e->getNextElementSibling();
  }
}

void Group::readXML() {
  // read XML
  shared_ptr<DOMParser> parser=DOMParser::create();
  shared_ptr<DOMDocument> doc=parser->parse(fileName);

  if(E(doc->getDocumentElement())->getTagName()!=OPENMBV%"Group")
    throw runtime_error("The root element must be of type {"+OPENMBV.getNamespaceURI()+"}Group");

  // read XML using OpenMBVCppInterface
  initializeUsingXML(doc->getDocumentElement());
}

void Group::write(bool writeXMLFile, bool writeH5File) {
  // use element name as base filename if fileName was not set
  if(fileName.empty()) fileName=name+".ombvx";

  if(writeH5File) {
    auto h5FileName=fileName.parent_path()/(fileName.stem().string()+".ombvh5");
    // This call will block until the h5 file can we opened for writing.
    // That is why we call it before calling writeXML.
    // This way the XML file will always be in sync with the H5 file since both use the same lock when the files are written.
    hdf5File=std::make_shared<H5::File>(h5FileName, H5::File::writeWithRename);
  }
  // now write the XML file (the H5 file is locked currently)
  if(writeXMLFile)
    writeXML();
  // now walk all objects and createw the corresponding groups/datasets in the H5 file
  if(writeH5File) {
    hdf5Group=hdf5File.get();
    for(auto & i : object)
      i->createHDF5File();
  }
  // Creating dataset is finished now and enableSWMR can be called to unblock the H5 file.
  // This is done explicity by the caller using Group::enableSWMR if wanted.
}

void Group::enableSWMR() {
  boost::filesystem::rename(getPreSWMRFileName(fileName), fileName);
  ombvxRenameNeeded = false;
  hdf5File->enableSWMR(); // this will unblock the h5 file
}

void Group::flushIfRequested() {
  hdf5File->flushIfRequested();
}

void Group::refresh() {
  hdf5File->refresh();
}

void Group::requestFlush() {
  if(hdf5File)
    hdf5File->requestFlush();
}

void Group::read() {
  std::shared_ptr<Group> p=parent.lock();
  // check if a corresponding H5 file exists, if yes ...
  string h5FileName(getFileName().substr(0,getFileName().length()-6)+".ombvh5");
  if(boost::filesystem::exists(h5FileName)) {
    // ... open the H5 file for reading. This will block the H5 file for writers.
    hdf5Group=nullptr;
    if(!p) {
      try {
        // this call will block until the h5 file can we opened for reading.
        // that is why we do it before calling readXML. This way readXML is also not read while a writer is active
        hdf5File=std::make_shared<H5::File>(h5FileName, H5::File::read, closeRequestCallback, refreshCallback);
        hdf5Group=hdf5File.get();
      }
      catch(...) {
        msg(Debug)<<"Unable to open the HDF5 File '"<<h5FileName<<"'. Using 0 for all data."<<endl;
      }
    }
  }

  // now read the XML file (the H5 file is currently locked for writer)
  readXML();

  if(getEnvironment() && boost::filesystem::exists(h5FileName))
    throw runtime_error("This XML file is an environment file but a corresponding H5 file exists!");
  if(!getEnvironment())
  {
    // now the structure is read from the XML file and we can open the corresponding H5 groups/datasets
    if(p) {
      try {
        if(!getEnvironment())
          hdf5Group=p->hdf5Group->openChildObject<H5::Group>(name);
      }
      catch(...) {
        msg(Debug)<<"Unable to open the HDF5 Group '"<<name<<"'. Using 0 for all data."<<endl;
      }
    }
    if(hdf5Group)
      for(auto & i : object)
        i->openHDF5File();
  }
}

}
