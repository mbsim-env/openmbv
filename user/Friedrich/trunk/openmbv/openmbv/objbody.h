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

#ifndef _OBJBODY_H_
#define _OBJBODY_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTextureCoordinate2.h>
#include <H5Cpp.h>

class ObjBody : public RigidBody {
  Q_OBJECT
  protected:
    enum Normals {
      fromObjFile,
      flat,
      smooth,
      smoothIfLessBarrier
    };
    enum Outline {
      none,
      calculate,
      fromFile
    };
    SoCoordinate3 *v;
    SoTextureCoordinate2 *t;
    SoNormal *n;
    class MtlMapGroup {
      public:
        MtlMapGroup();
        SoIndexedFaceSet *f;
        SoIndexedLineSet *ol;
        SoNormal *n;
        SoMaterial *mat;
        SoTexture2 *map;
    };
    double smoothBarrier;
    void readMtlLib(const std::string& mtlFile, std::map<QString, SoMaterial*>& material);
    void readMapLib(const std::string& mtlFile, std::map<QString, SoTexture2*>& map_);
  public:
    ObjBody(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
};

#endif
