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

#ifndef _OPENMBVGUI_EDGECALCULATION_H_
#define _OPENMBVGUI_EDGECALCULATION_H_

#include <fmatvec/atom.h>
#include <QtCore/QObject>
#include <vector>
#pragma push_macro("NDEBUG") // reworkaround some debugging bug in Coin3D
#define NDEBUG
#include <Inventor/nodes/SoCoordinate3.h>
#pragma pop_macro("NDEBUG")
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <QtCore/QReadWriteLock>

namespace OpenMBVGUI {

class EdgeCalculation : public QObject, virtual public fmatvec::Atom {
  Q_OBJECT
  public:
    /** Collect the data to be edge calculated from grp.
     * This function must be called from the main Coin thread and is very fast.
     * After this function the function preproces must be called exactly ones!
     * If useCache is true the function preproces uses a global, program wide, cache. */
    EdgeCalculation(SoGroup *grp, bool useCache=true);

    /** descructor */
    ~EdgeCalculation();

    /** Preproces the data collected with collectData.
     * This function is very time consuming for new data in grp from collectData.
     * For data in grp (in constructor) beeing alreay preprocessed it is cached if cache (in constructor) is true.
     * This function is thread safe in all cases!!! */
    void preproces(const std::string &fullName, bool printMessage=false);

    /** calculate the crease edges using the crease angle creaseAngle.
     * Before this funciton the function preproces must be called exactly ones!
     * This function is not very time consuming but is thread!!! */
    void calcCreaseEdges(const double creaseAngle);

    /** calculate the boundary edges.
     * Before this funciton the function preproces must be called exactly ones!
     * This function is not very time consuming but is thread!!! */
    void calcBoundaryEdges();

    /** calculate the shilouette edges using the given view normal n.
     * Before this funciton the function preproces must be called exactly ones!
     * This function is not very time consuming but is thread!!!
     * It must be called with the current normal n each time the view rotation changes. */
    void calcShilouetteEdges(const SbVec3f &n);

    /** return the coordinates used bey get*Edges().
     * Before this funciton the function preproces must be called exactly ones!
     * NOTE: Adding the coordinates to the scene graph must be done in the main Cion thread! */
    SoCoordinate3* getCoordinates() {
      return preData.coord;
    }

    /** return the crease edge face-set.
     * Before this funciton the function calcCreaseEdges must be called exactly ones!
     * NOTE: Adding the face-set to the scene graph must be done in the main Cion thread!
     * The coordinates used by this face-set are the ones return by getCoordinates */
    SoIndexedLineSet* getCreaseEdges() {
      return creaseEdges;
    }

    /** return the boundary edge face-set.
     * Before this funciton the function calcBoundaryEdges must be called exactly ones!
     * NOTE: Adding the face-set to the scene graph must be done in the main Cion thread!
     * The coordinates used by this face-set are the ones return by getCoordinates */
    SoIndexedLineSet* getBoundaryEdges() {
      return boundaryEdges;
    }

    /** return the shilouette edge face-set.
     * Before this funciton the function calcShilouetteEdges must be called exactly ones!
     * NOTE: Adding the face-set to the scene graph must be done in the main Cion thread!
     * The coordinates used by this face-set are the ones return by getCoordinates */
    SoIndexedLineSet* getShilouetteEdges() {
      return shilouetteEdges;
    }

  private:
    bool useCache;
    static void triangleCB(void *data, SoCallbackAction *action, const SoPrimitiveVertex *vp1, const SoPrimitiveVertex *vp2, const SoPrimitiveVertex *vp3);
    static SbVec3f v13OrthoTov12(SbVec3f v1, SbVec3f v2, SbVec3f v3);

    // set by collectData
    SoGroup *grp; // the scene group from which the data originates (This is used as key for the cache)
    std::vector<SbVec3f> *vertex; // all vertices from grp: each block of three points forms one triangle (allocated in collectData and freed in preproces)

    // set by preproces
    struct EdgeIndexFacePlaneVec {
      int vai, vbi; // index in coord of the vertex at the edgle line begin and end
      std::vector<SbVec3f> fpv; // vector in the face plane ortho to the edge (vb-va) (one vector for each face sharing this edge)
    };
    struct PreprocessedData { // preprocesses/cached data
      SoCoordinate3 *coord; // the coordinates for the face-sets (allocated in preproces and never freed, since the cache uses it)
      std::vector<EdgeIndexFacePlaneVec> *edgeIndFPV; // a 1D array for all edges (allocated in preproces and never freed, since the cache uses it)
      QReadWriteLock *calcLock; // is write locked until the calculation is running
    } preData;

    // set by calcCreaseEdges
    SoIndexedLineSet *creaseEdges;

    // set by calcBoundaryEdges
    SoIndexedLineSet *boundaryEdges;

    // set by calcShilouetteEdges
    SoIndexedLineSet *shilouetteEdges;

  signals:
    void statusBarShowMessage(const QString &message, int timeout=0);
};

}

#endif
