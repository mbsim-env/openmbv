#include <amviscppinterface/group.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <H5Cpp.h>

using namespace AMVis;
using namespace std;

Group::Group(const std::string& name_) : Object(name_), separateFile(false) {
}

void Group::addObject(Object* newObject) {
  for(int i=0; i<object.size(); i++)
    assert(object[i]->name!=newObject->name);
  object.push_back(newObject);
  newObject->parent=this;
}

void Group::writeXMLFile(ofstream& xmlFile, const string& indent) {
  xmlFile<<indent<<"<Group name=\""<<name<<"\">"<<endl;
    for(int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(xmlFile, indent+"  ");
  xmlFile<<indent<<"</Group>"<<endl;
}

void Group::createHDF5File() {
  hdf5Group=new H5::Group(parent->hdf5Group->createGroup(name));
  for(int i=0; i<object.size(); i++)
    object[i]->createHDF5File();
}

void Group::initialize() {
  ofstream xmlFile((name+".amvis.xml").c_str());
  xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<endl;
  xmlFile<<"<Group name=\""<<name<<"\" xmlns=\"http://www.amm.mw.tum.de/AMVis\">"<<endl;
    for(int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(xmlFile, "  ");
  xmlFile<<"</Group>"<<endl;
  xmlFile.close();

  hdf5Group=(H5::Group*)new H5::H5File(name+".amvis.h5", H5F_ACC_TRUNC);
  for(int i=0; i<object.size(); i++)
    object[i]->createHDF5File();
}
