#include <amviscppinterface/rigidbody.h>
#include <iostream>
#include <fstream>
#include <amviscppinterface/group.h>
#include <H5Cpp.h>

using namespace std;
using namespace AMVis;

RigidBody::RigidBody() : Body(),
  initialTranslation(3, 0),
  initialRotation(3, 0),
  scaleFactor(1),
  data(0) {
}

RigidBody::~RigidBody() {
  if(!hdf5LinkBody && data) delete data;
}

void RigidBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  Body::writeXMLFile(xmlFile, indent);
  xmlFile<<indent<<"<initialTranslation>["<<initialTranslation[0]<<";"
                                       <<initialTranslation[1]<<";"
                                       <<initialTranslation[2]<<"]</initialTranslation>"<<endl;
  xmlFile<<indent<<"<initialRotation>["<<initialRotation[0]<<";"
                                    <<initialRotation[1]<<";"
                                    <<initialRotation[2]<<"]</initialRotation>"<<endl;
  xmlFile<<indent<<"<scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
}

void RigidBody::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("x");
    columns.push_back("y");
    columns.push_back("z");
    columns.push_back("alpha");
    columns.push_back("beta");
    columns.push_back("gamma");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}
