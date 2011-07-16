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
#include "edgecalculation.h"
#include <Inventor/SoPrimitiveVertex.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/SbVec2i32.h>
#include <iostream>
#include <map>

using namespace std;

// HELPER CLASSES

// compare two SbVec3f objects
class SbVec3fComp {
  public:
    SbVec3fComp(double eps_) : eps(eps_) {}
    // return true if a<b
    bool operator()(const SbVec3f& a, const SbVec3f& b) const {
      double eps=0;
      if(fabs(a[0]-b[0])<=eps)
        if(fabs(a[1]-b[1])<=eps)
          if(a[2]<b[2] && fabs(a[2]-b[2])>eps)
            return true;
          else
            return false;
        else if(a[1]<b[1])
          return true;
        else
          return false;
      else if(a[0]<b[0])
        return true;
      else
        return false;
    }
  private:
    // eps value to thread to points as equal
    double eps;
};

// compare two SbVec2i32 objects
class SbVec2i32Comp {
  public:
    SbVec2i32Comp() {}
    // return true if a<b
    bool operator()(const SbVec2i32& a, const SbVec2i32& b) const {
      if(a[0]==b[0])
        if(a[1]<b[1])
          return true;
        else
          return false;
      else if(a[0]<b[0])
        return true;
      else
        return false;
    }
};

// A BSP tree analog to SbBSPTree (a Inventor extension of Coin) but faster and of arbitary type using templates
// Element: the element to be paritioned
// ElementComp: A class to compare two Element's using bool ElementComp::operator()(const ElementComp& a, const ElementComp& b);
template<class Element, class ElementComp>
class BSPTree {
  public:
    // create a BSP
    BSPTree() : ele2Index(), index2Ele(NULL) {} // using a default ElementComp object
    BSPTree(ElementComp comp) : ele2Index(comp), index2Ele(NULL) {} // using a given ElementComp object
    //descructor
    ~BSPTree() {
      delete[]index2Ele;
    }
    // add e to BSP tree an return the index of e
    unsigned int addPoint(const Element& e) {
      return ele2Index.insert(pair<Element, int>(e, ele2Index.size())).first->second;
    }
    // get number of points in BSP
    unsigned int numPoints() {
      return ele2Index.size();
    }
    // get a pointer to all vertices sorted by ascendent index.
    // The returned pointer gets invalid after a furthermore call to this function, addPoint or destructor.
    Element* getPointsArrayPtr() {
      delete[]index2Ele;
      index2Ele=new Element[ele2Index.size()];
      for(typename map<Element, int>::iterator i=ele2Index.begin(); i!=ele2Index.end(); i++)
        index2Ele[i->second]=i->first;
      return index2Ele;
    }
  private:
    // a map for all added points: sorted by the points: addPoint adds to this funciton in log(N)
    map<Element, int, ElementComp> ele2Index;
    // the reverse map of ele2Index: calculated only by getPointsArrayPtr
    Element *index2Ele;
};

// SoCoordinate3 which used the SbVec3f data from a BSPTree as points.
// The BSPTree is delete if the this object is deleted.
class SoCoordinate3FromBSPTree : public SoCoordinate3 {
  public:
    SoCoordinate3FromBSPTree(BSPTree<SbVec3f, SbVec3fComp> *bsp_) : bsp(bsp_) {
      point.setValuesPointer(bsp->numPoints(), bsp->getPointsArrayPtr());
    }
    ~SoCoordinate3FromBSPTree() {
      delete bsp;
    }
  private:
    BSPTree<SbVec3f, SbVec3fComp> *bsp;
};



// EDGE CALCULATION

void EdgeCalculation::triangleCB(void *data, SoCallbackAction *action, const SoPrimitiveVertex *vp1, const SoPrimitiveVertex *vp2, const SoPrimitiveVertex *vp3) {
  vector<SbVec3f> *vertex=(vector<SbVec3f>*)data;
  // get coordinates of points
  SbVec3f v1=vp1->getPoint();
  SbVec3f v2=vp2->getPoint();
  SbVec3f v3=vp3->getPoint();
  // convert coordinates to frame of the action
  SbMatrix mm=action->getModelMatrix();
  mm.multVecMatrix(v1, v1);
  mm.multVecMatrix(v2, v2);
  mm.multVecMatrix(v3, v3);
  // save points
  vertex->push_back(v1);
  vertex->push_back(v2);
  vertex->push_back(v3);
}

EdgeCalculation::EdgeCalculation(SoGroup *grp_, bool useCache_) {
  // initialize
  grp=grp_;
  useCache=useCache_;
  vertex=new vector<SbVec3f>;
  preData.coord=NULL;
  preData.edgeIndFPV=NULL;
  preData.calcLock=NULL;
  if(useCache) {
    preData.calcLock=new QReadWriteLock;
    preData.calcLock->lockForWrite();
  }
  creaseEdges=NULL;
  boundaryEdges=NULL;
  shilouetteEdges=NULL;
  // get all triangles
  SoCallbackAction cba;
  cba.addTriangleCallback(SoShape::getClassTypeId(), triangleCB, vertex);
  cba.apply(grp);
}

EdgeCalculation::~EdgeCalculation() {
  if(!useCache)
    delete preData.edgeIndFPV;
}

void EdgeCalculation::preproces(bool printMessage) {
  static map<SoGroup*, PreprocessedData> myCache;
  static QReadWriteLock mapRWLock;
  pair<map<SoGroup*, PreprocessedData>::iterator, bool> ins;
  if(useCache) {
    mapRWLock.lockForWrite(); // STL map is not thread safe => serialize
    ins=myCache.insert(pair<SoGroup*, PreprocessedData>(grp, preData)); // if not exist, add empty preData
    mapRWLock.unlock();
  }
  if(!useCache || ins.second) {
    // ADD A NEW ELEMENT
    if(printMessage)
      cout<<"Calculating edges!"<<endl;

    // CALCULATE
    preData.edgeIndFPV=new vector<EdgeIndexFacePlaneVec>; // is never freed, since the data is cached forever
    BSPTree<SbVec3f, SbVec3fComp> *uniqVertex=new BSPTree<SbVec3f, SbVec3fComp>(SbVec3fComp(0)); // a 3D float space paritioning for all vertex: allocate dynamically, since the points shared by preData.coords
    BSPTree<SbVec2i32, SbVec2i32Comp> edge; // a 2D interger space paritioning for all edges
    // build preData.edges struct from vertex
    for(unsigned int i=0; i<vertex->size()/3; i++) {
      // get points from vertex vector
      SbVec3f v1=(*vertex)[3*i+0];
      SbVec3f v2=(*vertex)[3*i+1];
      SbVec3f v3=(*vertex)[3*i+2];
      // add point and get point index
      unsigned int v1i=uniqVertex->addPoint(v1);
      unsigned int v2i=uniqVertex->addPoint(v2);
      unsigned int v3i=uniqVertex->addPoint(v3);
      // add edge and get edge index
      #define addEdge(i,j) addPoint(SbVec2i32(i<j?i:j,i<j?j:i)) // smaller index first, larger index second
      unsigned int e1i=edge.addEdge(v1i, v2i);
      unsigned int e2i=edge.addEdge(v2i, v3i);
      unsigned int e3i=edge.addEdge(v3i, v1i);
      #undef addEdge
      // add vai,vbi,fpv[...]
      EdgeIndexFacePlaneVec *x;
      #define expand(ei) if(ei>=preData.edgeIndFPV->size()) preData.edgeIndFPV->resize(ei+1);
      //expand size; get new/current element; set vai;    set vbi;    append fpv;
      expand(e1i); x=&(*preData.edgeIndFPV)[e1i];  x->vai=v1i; x->vbi=v2i; x->fpv.push_back(v13OrthoTov12(v1, v2, v3));
      expand(e2i); x=&(*preData.edgeIndFPV)[e2i];  x->vai=v2i; x->vbi=v3i; x->fpv.push_back(v13OrthoTov12(v2, v3, v1));
      expand(e3i); x=&(*preData.edgeIndFPV)[e3i];  x->vai=v3i; x->vbi=v1i; x->fpv.push_back(v13OrthoTov12(v3, v1, v2));
      #undef expand
    }
    delete vertex; // is no longer required and was allocate in getEdgeData
    // shift vertex points from BSP to preData.coord->point
    preData.coord=new SoCoordinate3FromBSPTree(uniqVertex);

    if(useCache) {
      mapRWLock.lockForWrite(); // STL map is not thread safe => serialize
      preData.calcLock->unlock(); // calculation is finished
      ins.first->second=preData; // set preprocessed data in cache
      mapRWLock.unlock();
    }
    return;
  }

  // GET AN EXISTING ELEMENT
  if(printMessage)
    cout<<"Use cached data! Waiting for cached data to get ready!"<<endl;
  
  delete preData.calcLock; // is not needed
  ins.first->second.calcLock->lockForRead(); // wait until tha cache entry has finished calculation
  ins.first->second.calcLock->unlock(); // unlock
  mapRWLock.lockForRead(); // STL map is not thread safe => serialize
  preData=ins.first->second; // set data from cache
  mapRWLock.unlock();
}

void EdgeCalculation::calcCreaseEdges(const double creaseAngle) {
  if(preData.coord==NULL) return;

  creaseEdges=new SoIndexedLineSet;
  int nr=0;
  for(unsigned int i=0; i<preData.edgeIndFPV->size(); i++) {
    // only draw crease edge if two faces belongs to this edge
    if((*preData.edgeIndFPV)[i].fpv.size()==2) {
      int vai=(*preData.edgeIndFPV)[i].vai; // index of edge start
      int vbi=(*preData.edgeIndFPV)[i].vbi; // index of edge end
      // draw crease edge if angle between fpv[0] and fpv[1] is < pi-creaseAngle
      if((*preData.edgeIndFPV)[i].fpv[0].dot((*preData.edgeIndFPV)[i].fpv[1])>-cos(creaseAngle)) {
        creaseEdges->coordIndex.set1Value(nr++, vai);
        creaseEdges->coordIndex.set1Value(nr++, vbi);
        creaseEdges->coordIndex.set1Value(nr++, -1);
      }
    }
  }
}

void EdgeCalculation::calcBoundaryEdges() {
  if(preData.coord==NULL) return;

  boundaryEdges=new SoIndexedLineSet;
  int nr=0;
  for(unsigned int i=0; i<preData.edgeIndFPV->size(); i++) {
    // draw boundary edge if only one face belongs to this edge
    if((*preData.edgeIndFPV)[i].fpv.size()==1) {
      boundaryEdges->coordIndex.set1Value(nr++, (*preData.edgeIndFPV)[i].vai);
      boundaryEdges->coordIndex.set1Value(nr++, (*preData.edgeIndFPV)[i].vbi);
      boundaryEdges->coordIndex.set1Value(nr++, -1);
    }
  }
}

void EdgeCalculation::calcShilouetteEdges(const SbVec3f &n) {
  if(preData.coord==NULL) return;

  shilouetteEdges=new SoIndexedLineSet;
  int nr=0;
  for(unsigned int i=0; i<preData.edgeIndFPV->size(); i++) {
    // only draw shilouette edge if two faces belongs to this edge
    if((*preData.edgeIndFPV)[i].fpv.size()==2) {
      int vai=(*preData.edgeIndFPV)[i].vai; // index of edge start
      int vbi=(*preData.edgeIndFPV)[i].vbi; // index of edge end
      SbVec3f v12=preData.coord->point[vbi]-preData.coord->point[vai]; // edge vector
      SbVec3f n0=v12.cross((*preData.edgeIndFPV)[i].fpv[0]); // normal of face 0
      SbVec3f n1=(*preData.edgeIndFPV)[i].fpv[1].cross(v12); // normal of face 1
      // draw shilouette edge if the face normals to different screen z directions (one i z+ one in z-)
      if(n0.dot(n)*n1.dot(n)<=0) {
        shilouetteEdges->coordIndex.set1Value(nr++, vai);
        shilouetteEdges->coordIndex.set1Value(nr++, vbi);
        shilouetteEdges->coordIndex.set1Value(nr++, -1);
      }
    }
  }
}

// return v13 in direction ortogonal to v12 (returned vec is normalized)
SbVec3f EdgeCalculation::v13OrthoTov12(SbVec3f v1, SbVec3f v2, SbVec3f v3) {
  SbVec3f v12=v2-v1;
  SbVec3f v13=v3-v1;
  SbVec3f ret=v13-v13.dot(v12)/v12.dot(v12)*v12;
  ret.normalize();
  return ret;
}
