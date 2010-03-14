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

#ifndef _UTILS_H_
#define _UTILS_H_

#include <QIcon>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/SbRotation.h>
#include <string>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/SoPrimitiveVertex.h>
#include <Inventor/SbBSPTree.h>
#include <Inventor/lists/SbVec3fList.h>
#include <GL/glu.h>

/** Utilitiy class */
class Utils {
  private:
    static bool init;

    static void triangleCB(void *data, SoCallbackAction *action, const SoPrimitiveVertex *vp1, const SoPrimitiveVertex *vp2, const SoPrimitiveVertex *vp3);
    struct EI2VINI {
      int vai, vbi; // index of the vertex at the line begin and end
      std::vector<int> ni; // index of all normal vectors of faces, which join this edge
    };

    // tess
    static GLenum tessType;
    static int tessNumVertices;
    static SoTriangleStripSet *tessTriangleStrip;
    static SoIndexedFaceSet *tessTriangleFan;
    static SoCoordinate3 *tessCoord;
    static void tessBeginCB(GLenum type, void *data);
    static void tessVertexCB(GLdouble *vertex);
    static void tessEndCB(void);

  public:
    static void initialize();

    /** Use QIconCached(filename) instead of QIcon(filename) everywhere
     * to cache the parsing of e.g. SVG files. This lead to a speedup
     * (at app init) by a factor of 11 in my test case. */
    static const QIcon& QIconCached(const QString& filename);
    
    /** Use SoDBreadAllCached(filename) instead of SoDBreadAll(filename) everywhere
     * to cache the iv-file parsing and scene generation */
    static SoSeparator* SoDBreadAllCached(const std::string &filename);

    /** string to vector */
    static std::vector<double> toVector(std::string str);
    static std::vector<std::vector<double> > toMatrix(std::string str);

    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale);
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble) { SoScale *scale; return soFrame(size, offset, pickBBoxAble, scale); }
    static SbRotation cardan2Rotation(const SbVec3f& c);
    static SbVec3f rotation2Cardan(const SbRotation& r);

    // calculate crease edges
    struct Edges {
      friend class Utils;
      private:
        SbBSPTree vertex; // a 3D float space paritioning for all vertex
        SbBSPTree edge; // a 2D interger space paritioning for all edges (ABUSES the class SbBSPTree)
        SbVec3fList normal; // a 1D array for the normals of all faces
        std::vector<EI2VINI> ei2vini; // a 1D array for all edges
    };
    /** Use preCalculateEdgesCached(...) instead of preCalculateEdges(...) everywhere
     * to cache the calculte of edges */
    static SoCoordinate3* preCalculateEdgesCached(SoGroup *grp, Edges *edges);
    static SoCoordinate3 *preCalculateEdges(SoGroup *sep, Edges *edges);
    static SoIndexedLineSet* calculateCreaseEdges(const double creaseAngle, const Edges *edges);
    static SoIndexedLineSet* calculateBoundaryEdges(const Edges *edges);
    static SoIndexedLineSet* calculateShilouetteEdge(const SbVec3f &n, const Edges *edges);

    // tess // convenience
    static GLUtesselator *tess;
};

#endif
