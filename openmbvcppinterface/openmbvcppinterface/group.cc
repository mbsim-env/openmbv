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
#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/body.h>
#include <openmbvcppinterface/objectfactory.h>
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include <hdf5serie/file.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

static string dirOfTopLevelFile(Group *grp);

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Group, OPENMBV%"Group")

Group::Group() : expandStr("true") {
}

Group::~Group() = default;

void Group::addObject(const shared_ptr<Object>& newObject) {
  if(newObject->name.empty()) throw runtime_error("object to add must have a name");
  for(auto & i : object)
    if(i->name==newObject->name)
      throw runtime_error("A object of name "+i->name+" already exists.");
  object.push_back(newObject);
  newObject->parent=shared_from_this();
}

string Group::getFullName(bool includingFileName, bool stopAtSeparateFile) {
  std::shared_ptr<Group> p=parent.lock();
  if(p) {
    if(separateFile && stopAtSeparateFile)
      return fileName;
    else
      return p->getFullName(includingFileName, stopAtSeparateFile)+"/"+name;
  }
  else
    return !includingFileName || fileName.empty() ? name : fileName;
}

DOMElement *Group::writeXMLFile(DOMNode *parent) {
  if(!separateFile) {
    DOMElement *e=Object::writeXMLFile(parent);
    E(e)->setAttribute("expand", expandStr);
    for(auto & i : object)
      i->writeXMLFile(e);
  }
  else {
    // use the fullName as file name of a separateFile Group with '/' replaced by '.'
    string fullName=getFullName();
    for(char & i : fullName) if(i=='/') i='.';
    // create link (embed) in current xml file
    DOMDocument *doc=parent->getNodeType()==DOMNode::DOCUMENT_NODE ? static_cast<DOMDocument*>(parent) : parent->getOwnerDocument();
    DOMElement *inc = D(doc)->createElement(XINCLUDE%"include");
    parent->insertBefore(inc, nullptr);
    E(inc)->setAttribute("href", fullName+".ombvx");
    fileName=dirOfTopLevelFile(this)+fullName+".ombvx";
    // create new xml file and write to it till now
    // use the directory of the topLevelFile and the above fullName
    shared_ptr<DOMParser> parser=DOMParser::create();
    shared_ptr<DOMDocument> xmlFile=parser->createDocument();
      DOMElement *e=Object::writeXMLFile(xmlFile.get());
      E(e)->setAttribute("expand", expandStr);
      for(auto & i : object)
        i->writeXMLFile(e);
    DOMParser::serialize(xmlFile.get(), fileName);
  }
  return nullptr;
}

void Group::createHDF5File() {
  std::shared_ptr<Group> p=parent.lock();
  if(!separateFile) {
    hdf5Group=p->hdf5Group->createChildObject<H5::Group>(name)();
    for(auto & i : object)
      if(!i->getEnvironment()) i->createHDF5File();
  }
  else {
    string fullName=getFullName();
    for(char & i : fullName) if(i=='/') i='.';
    // create link in current h5 file
    p->hdf5Group->createExternalLink(name, make_pair(boost::filesystem::path(fullName+".ombvh5"), string("/")));
    // create new h5 file and write to in till now
    // use the directory of the topLevelFile and the above fullName
    fileName=dirOfTopLevelFile(this)+fullName+".ombvx";
    hdf5File=std::make_shared<H5::File>(fileName.substr(0,fileName.length()-6)+".ombvh5", H5::File::write);
    hdf5Group=hdf5File.get();
    for(auto & i : object)
      if(!i->getEnvironment()) i->createHDF5File();
  }
}

void Group::openHDF5File() {
  hdf5Group=nullptr;
  std::shared_ptr<Group> p=parent.lock();
  if(!p)
    throw runtime_error("mfmfxxxxa");
  else {
    try {
      if(!getEnvironment())
        hdf5Group=p->hdf5Group->openChildObject<H5::Group>(name);
    }
    catch(...) {
      msg(Warn)<<"Unable to open the HDF5 Group '"<<name<<"'"<<endl;
    }
  }
  if(hdf5Group)
    for(auto & i : object)
      i->openHDF5File();
}

void Group::writeXML() {
  separateFile=true;
  // write .ombvx file
  shared_ptr<DOMParser> parser=DOMParser::create();
  shared_ptr<DOMDocument> xmlFile=parser->createDocument();
  DOMElement *parent=Object::writeXMLFile(xmlFile.get());
  E(parent)->setAttribute("expand", expandStr);
  for(auto & i : object)
    i->writeXMLFile(parent);
  DOMParser::serialize(xmlFile.get(), fileName);
}

void Group::terminate() {
  for(auto & i : object)
    i->terminate();
  if(separateFile)
    delete static_cast<H5::File*>(hdf5Group);
  hdf5Group=nullptr;
}

void Group::initializeUsingXML(DOMElement *element) {
  Object::initializeUsingXML(element);
  if(E(element)->hasAttribute("expand") && 
     (E(element)->getAttribute("expand")=="false" || E(element)->getAttribute("expand")=="0"))
    setExpand(false);
  DOMProcessingInstruction *ofn=E(element)->getFirstProcessingInstructionChildNamed("OriginalFilename");
  if(ofn) {
    setSeparateFile(true);
    fileName=X()%ofn->getData();
  }

  DOMElement *e;
  e=element->getFirstElementChild();
  while (e) {
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

  // read XML using OpenMBVCppInterface
  initializeUsingXML(doc->getDocumentElement());
}

string dirOfTopLevelFile(Group *grp) {
  // get directory of top level file
  string dir=grp->getTopLevelGroup()->getFileName();
  size_t pos=dir.find_last_of('/');
  if(pos!=string::npos)
    dir=dir.substr(0, pos+1);
  else
    dir="";
  return dir;
}

void Group::write(bool writeXMLFile, bool writeH5File) {
  // use element name as base filename if fileName was not set
  if(fileName.empty()) fileName=name+".ombvx";

  if(writeH5File) {
    string h5FileName=fileName.substr(0,fileName.length()-6)+".ombvh5";
    // This call will block until the h5 file can we opened for writing.
    // That is why we call it before calling writeXML.
    // This way the XML file will always be in sync with the H5 file since both use the same lock when the files are written.
    hdf5File=std::make_shared<H5::File>(h5FileName, H5::File::write);
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
  hdf5File->enableSWMR(); // this will unblock the h5 file
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
        hdf5File=std::make_shared<H5::File>(h5FileName, H5::File::read, closeRequestCallback);
        hdf5Group=hdf5File.get();
      }
      catch(...) {
        msg(Warn)<<"Unable to open the HDF5 File '"<<h5FileName<<"'"<<endl;
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
        msg(Warn)<<"Unable to open the HDF5 Group '"<<name<<"'"<<endl;
      }
    }
    if(hdf5Group)
      for(auto & i : object)
        i->openHDF5File();
  }
}

}
