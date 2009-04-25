#include <openmbvcppinterface/coilspring.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

CoilSpring::CoilSpring() : Body(),
  springRadius(1),
  crossSectionRadius(0.1),
  scaleFactor(1) {
}

void CoilSpring::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<CoilSpring name=\""<<name<<"\" name=\""<<expandStr<<"\">"<<endl;
    Body::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <numberOfCoils>"<<numberOfCoils<<"</numberOfCoils>"<<endl;
    xmlFile<<indent<<"  <springRadius>"<<springRadius<<"</springRadius>"<<endl;
    xmlFile<<indent<<"  <crossSectionRadius>"<<crossSectionRadius<<"</crossSectionRadius>"<<endl;
    xmlFile<<indent<<"  <scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
  xmlFile<<indent<<"</CoilSpring>"<<endl;
}

void CoilSpring::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("fromPoint x");
    columns.push_back("fromPoint y");
    columns.push_back("fromPoint z");
    columns.push_back("toPoint x");
    columns.push_back("toPoint y");
    columns.push_back("toPoint z");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}
