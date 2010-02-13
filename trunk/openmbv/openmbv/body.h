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

#ifndef _BODY_H_
#define _BODY_H_

#include "config.h"
#include "object.h"
#include "tinyxml.h"
#include <Inventor/sensors/SoFieldSensor.h>
#include <H5Cpp.h>
#include <QtGui/QActionGroup>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <GL/glu.h>
#include "IndexedTesselationFace.h"
#include <Inventor/SbBSPTree.h>
#include <Inventor/lists/SbVec3fList.h>
#include <Inventor/nodes/SoIndexedLineSet.h>

class Body : public Object {
  Q_OBJECT
  private:
    enum DrawStyle { filled, lines, points };
    SoDrawStyle *drawStyle;
    static bool existFiles;
    static Body *timeUpdater; // the body who updates the time string in the scene window
  public:
    Body(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
    static void frameSensorCB(void *data, SoSensor*);
    virtual QMenu* createMenu();
    virtual double update()=0; // return the current time
    void resetAnimRange(int numOfRows, double dt);
  public slots:
    void outLineSlot();
  protected slots:
    void drawMethodSlot(QAction* action);
  protected:
    SoSwitch *soOutLineSwitch;
    SoSeparator *soOutLineSep;
    QAction *outLine;
    QActionGroup *drawMethod;
    QAction *drawMethodPolygon, *drawMethodLine, *drawMethodPoint;
    friend void IndexedTesselationFace::changedCB(void *data, SoSensor*);



  // FROM NOW ONLY CONVENIENCE FUNCTIONS FOLLOW !!!
  protected:
    // tess // convenience
    static GLUtesselator *tess; // convenience
    static GLenum tessType; // convenience
    static int tessNumVertices; // convenience
    static SoTriangleStripSet *tessTriangleStrip; // convenience
    static SoIndexedFaceSet *tessTriangleFan; // convenience
    static SoCoordinate3 *tessCoord; // convenience
    static bool tessCBInit; // convenience
    static void tessBeginCB(GLenum type, void *data); // convenience
    static void tessVertexCB(GLdouble *vertex); // convenience
    static void tessEndCB(void); // convenience

    // string to vector
    static std::vector<double> toVector(std::string str); // convenience
    static std::vector<std::vector<double> > toMatrix(std::string str); // convenience

    // calculate crease edges
    struct Edges {
      SbBSPTree vertex;
      SbIntList faceVertex;
      SbVec3fList normal;
      SbIntList innerEdge;
      SoMFInt32 boundaryEdge;
    };
    static void triangleCB(void *data, SoCallbackAction *action, const SoPrimitiveVertex *vp1, const SoPrimitiveVertex *vp2, const SoPrimitiveVertex *vp3);
    SoCoordinate3 *preCalculateEdges(SoGroup *sep, Edges *edges);
    SoIndexedLineSet* calculateCreaseEdges(double creaseAngle, Edges *edges);
    SoIndexedLineSet* calculateBoundaryEdges(Edges *edges);

  public:
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble) { SoScale *scale; return soFrame(size, offset, pickBBoxAble, scale); } // convenience
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale); // convenience
    static SbRotation cardan2Rotation(const SbVec3f& c);
    static SbVec3f rotation2Cardan(const SbRotation& r);
};

#endif
