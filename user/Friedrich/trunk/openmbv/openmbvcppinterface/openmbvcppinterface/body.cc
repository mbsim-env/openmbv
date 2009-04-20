#include <openmbvcppinterface/body.h>
#include <iostream>
#include <fstream>
#include <H5Cpp.h>
#include <openmbvcppinterface/group.h>

using namespace std;
using namespace OpenMBV;

Body::Body() : Object(),
  hdf5LinkBody(0) {
}

void Body::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  if(hdf5LinkBody)
    xmlFile<<indent<<"<hdf5Link ref=\""<<getRelPathTo(hdf5LinkBody)<<"\"/>"<<endl;
}

void Body::createHDF5File() {
  if(!hdf5LinkBody)
    hdf5Group=new H5::Group(parent->hdf5Group->createGroup(name));
  else
    parent->hdf5Group->link(H5L_TYPE_SOFT, getRelPathTo(hdf5LinkBody), name);
}

std::string Body::getRelPathTo(Body* destBody) {
  // create relative path to destination
  string dest=destBody->getFullName();
  string src=getFullName();
  string reldest="";
  while(dest.substr(0,dest.find('/',1))==src.substr(0,src.find('/',1)))  {
    dest=dest.substr(dest.find('/',1));
    src=src.substr(src.find('/',1));
  }
  while((signed)src.find('/',1)>=0) {
    reldest=reldest+"../";
    src=src.substr(src.find('/',1));
  }
  reldest=reldest+dest.substr(1);
  return reldest;
}

void Body::terminate() {
}
