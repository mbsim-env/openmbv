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

#include "config.h"
#include "path.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoBaseColor.h>
#include "utils.h"
#include "openmbvcppinterface/path.h"

using namespace std;

Path::Path(OpenMBV::Object *obj, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Body(obj, h5Parent, parentItem, soParent) {
  OpenMBV::Path* p=(OpenMBV::Path*)obj;
  iconFile=":/path.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  //h5 dataset
  h5Data=new H5::VectorSerie<double>;
  if(h5Group) {
    h5Data->open(*h5Group, "data");
    int rows=h5Data->getRows();
    double dt;
    if(rows>=2) dt=h5Data->getRow(1)[0]-h5Data->getRow(0)[0]; else dt=0;
    resetAnimRange(rows, dt);
  }
  
  // create so
  SoBaseColor *col=new SoBaseColor;
  col->rgb.setValue(p->getColor()[0], p->getColor()[1], p->getColor()[2]);
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
