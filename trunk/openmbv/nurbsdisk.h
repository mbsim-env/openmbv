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

#ifndef _NURBSDISK_H_
#define _NURBSDISK_H_

#include "config.h"
#include "body.h"
#include "tinyxml.h"
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedNurbsSurface.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>
#include <QtGui/QMenu>

/**
 * \brief class for bodies with NURBS surface and primitive closure
 * \author Kilian Grundl
 * \author Raphael Missel
 * \author Thorsten Schindler
 * \date 2009-05-20 initial commit (Grundl / Missel / Schindler)
 */
class NurbsDisk : public Body {
  public:
    /** constructor */
    NurbsDisk(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);

    /** info string in spine extrusion pop-up menu */
    virtual QString getInfo();

  protected:
    /** NURBS surface */
    SoIndexedNurbsSurface *surface;

    /** primitive closures */
    SoIndexedFaceSet *back;
    SoIndexedFaceSet *side;

    /** local h5 data set copy */
    H5::VectorSerie<double> *h5Data;
  
    /** update method invoked at each time step */
    virtual double update();
};

#endif /* _NURBSDISK_H_ */

