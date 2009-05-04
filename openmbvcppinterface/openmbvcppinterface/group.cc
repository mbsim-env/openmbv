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

#include <openmbvcppinterface/group.h>
#include <hdf5serie/fileserie.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <H5Cpp.h>

using namespace OpenMBV;
using namespace std;

Group::Group() : Object(), separateFile(false) {
}

void Group::addObject(Object* newObject) {
  assert(newObject->name!="");
  for(int i=0; i<object.size(); i++)
    assert(object[i]->name!=newObject->name);
  object.push_back(newObject);
  newObject->parent=this;
}

void Group::writeXMLFile(ofstream& xmlFile, const string& indent) {
  if(!separateFile) {
    xmlFile<<indent<<"<Group name=\""<<name<<"\" expand=\""<<expandStr<<"\">"<<endl;
      for(int i=0; i<object.size(); i++)
        object[i]->writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"</Group>"<<endl;
  }
  else {
    string fullName=getFullName();
    for(int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link (embed) in current xml file
    xmlFile<<indent<<"<xi:include href=\""+fullName+".ombv.xml\"/>"<<endl;
    // create new xml file and write to it till now
    ofstream newxmlFile((fullName+".ombv.xml").c_str());
    newxmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<endl;
    newxmlFile<<"<Group name=\""<<name<<"\" xmlns=\""OPENMBVNS_"\""<<endl<<
                "  xmlns:xi=\"http://www.w3.org/2001/XInclude\">"<<endl;
      for(int i=0; i<object.size(); i++)
        object[i]->writeXMLFile(newxmlFile, "  ");
    newxmlFile<<"</Group>"<<endl;
    newxmlFile.close();
  }
}

void Group::createHDF5File() {
  if(!separateFile) {
    hdf5Group=new H5::Group(parent->hdf5Group->createGroup(name));
    for(int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
  else {
    string fullName=getFullName();
    for(int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link in current h5 file
    H5Lcreate_external((fullName+".ombv.h5").c_str(), "/",
                       parent->hdf5Group->getId(), name.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT);
    // create new h5 file land write to in till now
    hdf5Group=(H5::Group*)new H5::FileSerie(fullName+".ombv.h5", H5F_ACC_TRUNC);
    for(int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
}

void Group::initialize() {
  ofstream xmlFile((name+".ombv.xml").c_str());
  xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<endl;
  xmlFile<<"<Group name=\""<<name<<"\" xmlns=\""OPENMBVNS_"\""<<endl<<
           "  xmlns:xi=\"http://www.w3.org/2001/XInclude\">"<<endl;
    for(int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(xmlFile, "  ");
  xmlFile<<"</Group>"<<endl;
  xmlFile.close();

  hdf5Group=(H5::Group*)new H5::FileSerie(name+".ombv.h5", H5F_ACC_TRUNC);
  for(int i=0; i<object.size(); i++)
    object[i]->createHDF5File();
  hdf5Group->flush(H5F_SCOPE_GLOBAL);
}

void Group::terminate() {
  for(int i=0; i<object.size(); i++)
    object[i]->terminate();
  if(!separateFile)
    delete hdf5Group;
  else
    delete (H5::FileSerie*)hdf5Group;
  hdf5Group=0;
}
