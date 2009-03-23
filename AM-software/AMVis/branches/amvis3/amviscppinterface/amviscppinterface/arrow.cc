#include <amviscppinterface/arrow.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AMVis;

Arrow::Arrow() : Body(),
  diameter(1),
  headDiameter(0.5),
  headLength(0.75),
  type(toHead) {
}

void Arrow::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Arrow name=\""<<name<<"\">"<<endl;
    Body::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <diameter>"<<diameter<<"</diameter>"<<endl;
    xmlFile<<indent<<"  <headDiameter>"<<headDiameter<<"</headDiameter>"<<endl;
    xmlFile<<indent<<"  <headLength>"<<headLength<<"</headLength>"<<endl;
    string typeStr;
    switch(type) {
      case noHead: typeStr="noHead"; break;
      case fromHead: typeStr="fromHead"; break;
      case toHead: typeStr="toHead"; break;
      case bothHeads: typeStr="bothHeads"; break;
    }
    xmlFile<<indent<<"  <type>"<<typeStr<<"</type>"<<endl;
  xmlFile<<indent<<"</Arrow>"<<endl;
}

void Arrow::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("toPoint x");
    columns.push_back("toPoint x");
    columns.push_back("toPoint x");
    columns.push_back("delte x");
    columns.push_back("delte y");
    columns.push_back("delte z");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}
