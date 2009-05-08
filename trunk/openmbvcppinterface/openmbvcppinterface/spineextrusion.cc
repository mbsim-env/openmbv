#include <openmbvcppinterface/spineextrusion.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

SpineExtrusion::SpineExtrusion() : Body(),
  numberOfSpinePoints(0),
  contour(0),
  data(0), 
  staticColor(-1),
  minimalColorValue(0.),
  maximalColorValue(1.),
  scaleFactor(1) {
  }

  SpineExtrusion::~SpineExtrusion() {
    if(!hdf5LinkBody && data) delete data;
  }

void SpineExtrusion::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<SpineExtrusion name=\""<<name<<"\" expand=\""<<expandStr<<"\">"<<endl;
  Body::writeXMLFile(xmlFile, indent+"  ");
  if(contour) PolygonPoint::serializePolygonPointContour(xmlFile, indent+"  ", contour);
  xmlFile<<indent<<"  <minimalColorValue>"<<minimalColorValue<<"</minimalColorValue>"<<endl;
  xmlFile<<indent<<"  <maximalColorValue>"<<maximalColorValue<<"</maximalColorValue>"<<endl;
  xmlFile<<indent<<"  <scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
  xmlFile<<indent<<"  <color>"<<staticColor<<"</color>"<<endl;
  xmlFile<<indent<<"</SpineExtrusion>"<<endl;
}

void SpineExtrusion::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    for(int i=0;i<numberOfSpinePoints;i++) {
      columns.push_back("x"+numtostr(i));
      columns.push_back("y"+numtostr(i));
      columns.push_back("z"+numtostr(i));
      columns.push_back("twist"+numtostr(i));
    }
    data->create(*hdf5Group,"data",columns);
  }
}

