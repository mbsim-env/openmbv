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

#ifndef _PATH_H_
#define _PATH_H_

#include "config.h"
#include "body.h"
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>

class Path : public Body {
  Q_OBJECT
  protected:
    virtual double update();
    H5::VectorSerie<double> *h5Data;
    SoCoordinate3 *coord;
    SoLineSet *line;
    int maxFrameRead;
  public:
    Path(OpenMBV::Object* obj, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
    virtual QString getInfo();
};

#endif
