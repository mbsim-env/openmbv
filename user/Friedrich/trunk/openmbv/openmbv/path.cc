#include "config.h"
#include "path.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoBaseColor.h>

using namespace std;

Path::Path(TiXmlElement *element, H5::Group *h5Parent) : Body(element, h5Parent) {
  iconFile=":/path.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  //h5 dataset
  h5Data=new H5::VectorSerie<double>;
  if(h5Group) {
    h5Data->open(*h5Group, "data");
    int rows=h5Data->getRows();
    double dt;
    if(rows>=2) dt=h5Data->getRow(1)[0]-h5Data->getRow(0)[0]; else dt=0;
    resetAnimRange(rows, dt);
  }
  
  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"color");
  vector<double> color=toVector(e->GetText());

  // create so
  SoBaseColor *col=new SoBaseColor;
  col->rgb.setValue(color[0], color[1], color[2]);
  soSep->addChild(col);
  coord=new SoCoordinate3;
  soSep->addChild(coord);
  line=new SoLineSet;
  soSep->addChild(line);
  maxFrameRead=-1;
}

double Path::update() {
  if(h5Group==0) return 0;
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=h5Data->getRow(frame);
  for(int i=maxFrameRead+1; i<=frame; i++) {
    vector<double> data=h5Data->getRow(i);
    coord->point.set1Value(i, data[1], data[2], data[3]);
  }
  maxFrameRead=frame;
  line->numVertices.setValue(1+frame);

  return data[0];
}

QString Path::getInfo() {
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  float x, y, z;
  coord->point.getValues(frame)->getValue(x, y, z);
  return Body::getInfo()+
         QString("-----<br/>")+
         QString("<b>Position:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z);
}
