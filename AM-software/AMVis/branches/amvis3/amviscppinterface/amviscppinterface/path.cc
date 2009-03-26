#include <amviscppinterface/path.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AMVis;

Path::Path() : Body() {
}

void Path::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Path name=\""<<name<<"\">"<<endl;
    Body::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <color>["<<color[0]<<";"
                                 <<color[1]<<";"
                                 <<color[2]<<"]</color>"<<endl;
  xmlFile<<indent<<"</Path>"<<endl;
}

void Path::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("x");
    columns.push_back("y");
    columns.push_back("z");
    data->create(*hdf5Group,"data",columns);
  }
}
