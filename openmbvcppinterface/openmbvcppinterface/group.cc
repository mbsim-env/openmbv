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
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#ifdef HAVE_BOOST_FILE_LOCK
#  include <boost/interprocess/sync/file_lock.hpp>
#endif

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

static string dirOfTopLevelFile(Group *grp);

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Group, OPENMBV%"Group")

Group::Group() : Object(), expandStr("true"), separateFile(false) {
}

Group::~Group() {
}

void Group::addObject(shared_ptr<Object> newObject) {
  if(newObject->name=="") throw runtime_error("object to add must have a name");
  for(unsigned int i=0; i<object.size(); i++)
    if(object[i]->name==newObject->name)
      throw runtime_error("A object of name "+object[i]->name+" already exists.");
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
    return includingFileName==false || fileName.empty() ? name : fileName;
}

DOMElement *Group::writeXMLFile(DOMNode *parent) {
  if(!separateFile) {
    DOMElement *e=Object::writeXMLFile(parent);
    addAttribute(e, "expand", expandStr, "true");
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(e);
  }
  else {
    // use the fullName as file name of a separateFile Group with '/' replaced by '.'
    string fullName=getFullName();
    for(unsigned int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link (embed) in current xml file
    DOMDocument *doc=parent->getNodeType()==DOMNode::DOCUMENT_NODE ? static_cast<DOMDocument*>(parent) : parent->getOwnerDocument();
    DOMElement *inc = D(doc)->createElement(XINCLUDE%"include");
    parent->insertBefore(inc, NULL);
    E(inc)->setAttribute("href", fullName+".ombv.xml");
    fileName=dirOfTopLevelFile(this)+fullName+".ombv.xml";
    // create new xml file and write to it till now
    // use the directory of the topLevelFile and the above fullName
    shared_ptr<DOMParser> parser=DOMParser::create();
    shared_ptr<DOMDocument> xmlFile=parser->createDocument();
      DOMElement *e=Object::writeXMLFile(xmlFile.get());
      addAttribute(e, "expand", expandStr, "true");
      for(unsigned int i=0; i<object.size(); i++)
        object[i]->writeXMLFile(e);
    DOMParser::serialize(xmlFile.get(), fileName);
  }
  return 0;
}

void Group::createHDF5File() {
  std::shared_ptr<Group> p=parent.lock();
  if(!separateFile) {
    hdf5Group=p->hdf5Group->createChildObject<H5::Group>(name)();
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
  else {
    string fullName=getFullName();
    for(unsigned int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link in current h5 file
    p->hdf5Group->createExternalLink(name, make_pair(boost::filesystem::path(fullName+".ombv.h5"), string("/")));
    // create new h5 file and write to in till now
    // use the directory of the topLevelFile and the above fullName
    fileName=dirOfTopLevelFile(this)+fullName+".ombv.xml";
    hdf5File=std::make_shared<H5::File>(fileName.substr(0,fileName.length()-4)+".h5", H5::File::write);
    hdf5Group=hdf5File.get();
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
}

void Group::openHDF5File() {
  hdf5Group=NULL;
  std::shared_ptr<Group> p=parent.lock();
  if(!p) {
    try {
      hdf5File=std::make_shared<H5::File>(getFileName().substr(0,getFileName().length()-4)+".h5", H5::File::read);
      hdf5Group=hdf5File.get();
    }
    catch(...) {
      msg(Warn)<<"Unable to open the HDF5 File '"<<getFileName().substr(0,getFileName().length()-4)+".h5"<<"'"<<endl;
    }
  }
  else {
    try {
      hdf5Group=p->hdf5Group->openChildObject<H5::Group>(name);
    }
    catch(...) {
      msg(Warn)<<"Unable to open the HDF5 Group '"<<name<<"'"<<endl;
    }
  }
  if(hdf5Group)
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->openHDF5File();
}

void Group::writeXML() {
  separateFile=true;
  // write .ombv.xml file
    shared_ptr<DOMParser> parser=DOMParser::create();
    shared_ptr<DOMDocument> xmlFile=parser->createDocument();
    DOMElement *parent=Object::writeXMLFile(xmlFile.get());
    addAttribute(parent, "expand", expandStr, "true");
    for(unsigned int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(parent);
  DOMParser::serialize(xmlFile.get(), fileName);
}

void Group::writeH5() {
  string h5FileName=fileName.substr(0,fileName.length()-4)+".h5";
  hdf5File=std::make_shared<H5::File>(h5FileName, H5::File::write);
  hdf5Group=hdf5File.get();
  for(unsigned int i=0; i<object.size(); i++)
    object[i]->createHDF5File();
  hdf5File->flush();
}

void Group::terminate() {
  for(unsigned int i=0; i<object.size(); i++)
    object[i]->terminate();
  if(separateFile)
    delete static_cast<H5::File*>(hdf5Group);
  hdf5Group=0;
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

void Group::readH5() {
  openHDF5File();
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
  if(fileName=="") fileName=name+".ombv.xml";

#ifdef HAVE_BOOST_FILE_LOCK
  size_t size=fileName.find_last_of("/\\");
  int pos;
  if(size==string::npos) pos=-1; else pos=static_cast<int>(size);
  string lockFileName=fileName.substr(0, pos+1)+"."+fileName.substr(pos+1, fileName.length()-4)+".lock";

  // try to create file
  FILE *f=fopen(lockFileName.c_str(), "a");
  if(f) fclose(f);
  // try to lock file
  boost::interprocess::file_lock fileLock;
  bool isFileLocked=false;
  try {
    if(!getenv("OPENMBVCPPINTERFACE_NO_FILE_LOCK")) {
      boost::interprocess::file_lock fl(lockFileName.c_str());
      fileLock.swap(fl);
      isFileLocked=true;
      fileLock.lock();
    }
  }
  catch(const boost::interprocess::interprocess_exception &ex) {
    msg(Warn)<<"Unable to lock the file "<<lockFileName<<endl;
  }
#endif
  try {
    if(writeXMLFile) writeXML();
    if(writeH5File)  writeH5();
  }
  catch(...) {
#ifdef HAVE_BOOST_FILE_LOCK
    // unlock file
    if(isFileLocked)
      fileLock.unlock();
#endif
    throw;
  }
#ifdef HAVE_BOOST_FILE_LOCK
  // unlock file
  if(isFileLocked)
    fileLock.unlock();
#endif
}

void Group::read(bool readXMLFile, bool readH5File) {
#ifdef HAVE_BOOST_FILE_LOCK
  size_t size=fileName.find_last_of("/\\");
  int pos;
  if(size==string::npos) pos=-1; else pos=static_cast<int>(size);
  string lockFileName=fileName.substr(0, pos+1)+"."+fileName.substr(pos+1, fileName.length()-4)+".lock";

  // try to create file
  FILE *f=fopen(lockFileName.c_str(), "a");
  if(f) fclose(f);
  // try to lock file
  boost::interprocess::file_lock fileLock;
  bool isFileLocked=false;
  try {
    if(!getenv("OPENMBVCPPINTERFACE_NO_FILE_LOCK")) {
      boost::interprocess::file_lock fl(lockFileName.c_str());
      fileLock.swap(fl);
      isFileLocked=true;
      fileLock.lock_sharable();
    }
  }
  catch(const boost::interprocess::interprocess_exception &ex) {
    msg(Warn)<<"Unable to lock the file "<<lockFileName<<endl;
  }
#endif
  try {
    if(readXMLFile) readXML();
    if(readH5File)  readH5();
  }
  catch(...) {
#ifdef HAVE_BOOST_FILE_LOCK
    // unlock file
    if(isFileLocked)
      fileLock.unlock();
#endif
    throw;
  }
#ifdef HAVE_BOOST_FILE_LOCK
  // unlock file
  if(isFileLocked)
    fileLock.unlock_sharable();
#endif
}

}
