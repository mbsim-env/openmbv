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
  if(!separateFile) {
    xmlFile<<indent<<"<Group name=\""<<name<<"\">"<<endl;
      for(int i=0; i<object.size(); i++)
        object[i]->writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"</Group>"<<endl;
  }
  else {
    string fullName=getFullName();
    for(int i=0; i<fullName.length(); i++) if(fullName[i]=='/') fullName[i]='.';
    // create link (embed) in current xml file
    xmlFile<<indent<<"<pv:embed href=\""+fullName+".amvis.xml\"/>"<<endl;
    // create new xml file and write to it till now
    ofstream newxmlFile((fullName+".amvis.xml").c_str());
    newxmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<endl;
    newxmlFile<<"<Group name=\""<<name<<"\" xmlns=\"http://www.amm.mw.tum.de/AMVis\""<<endl<<
                "  xmlns:pv=\"http://hdf5serie.berlios.de/PV\">"<<endl;
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
    H5Lcreate_external((fullName+".amvis.h5").c_str(), "/",
                       parent->hdf5Group->getId(), name.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT);
    // create new h5 file land write to in till now
    hdf5Group=(H5::Group*)new H5::H5File(fullName+".amvis.h5", H5F_ACC_TRUNC);
    for(int i=0; i<object.size(); i++)
      object[i]->createHDF5File();
  }
}

void Group::initialize() {
  ofstream xmlFile((name+".amvis.xml").c_str());
  xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<endl;
  xmlFile<<"<Group name=\""<<name<<"\" xmlns=\"http://www.amm.mw.tum.de/AMVis\""<<endl<<
           "  xmlns:pv=\"http://hdf5serie.berlios.de/PV\">"<<endl;
    for(int i=0; i<object.size(); i++)
      object[i]->writeXMLFile(xmlFile, "  ");
  xmlFile<<"</Group>"<<endl;
  xmlFile.close();

  hdf5Group=(H5::Group*)new H5::H5File(name+".amvis.h5", H5F_ACC_TRUNC);
  for(int i=0; i<object.size(); i++)
    object[i]->createHDF5File();
}
