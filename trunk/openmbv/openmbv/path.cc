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

Path::Path(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent) : Body(obj, parentItem, soParent) {
  path=(OpenMBV::Path*)obj;
  iconFile=":/path.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  //h5 dataset
  int rows=path->getRows();
  double dt;
  if(rows>=2) dt=path->getRow(1)[0]-path->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);
  
  // create so
  SoBaseColor *col=new SoBaseColor;
  col->rgb.setValue(path->getColor()[0], path->getColor()[1], path->getColor()[2]);
  soSep->addChild(col);
  coord=new SoCoordinate3;
  soSep->addChild(coord);
  line=new SoLineSet;
  soSep->addChild(line);
  maxFrameRead=-1;
}

double Path::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=path->getRow(frame);
  for(int i=maxFrameRead+1; i<=frame; i++) {
    vector<double> data=path->getRow(i);
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
