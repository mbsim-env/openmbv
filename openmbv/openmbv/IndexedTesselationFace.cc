#include "config.h"
#include "IndexedTesselationFace.h"
#include <Inventor/nodes/SoIndexedFaceSet.h>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN // GL/glu.h includes windows.h on Windows -> avoid full header -> WIN32_LEAN_AND_MEAN
#endif
#include <GL/glu.h>
#include "body.h"

namespace OpenMBVGUI {

SO_NODE_SOURCE(IndexedTesselationFace);

void IndexedTesselationFace::initClass() {
  SO_NODE_INIT_CLASS(IndexedTesselationFace, SoGroup, "Group");
}

void IndexedTesselationFace::constructor() {
  SO_NODE_CONSTRUCTOR(IndexedTesselationFace);
  SO_NODE_DEFINE_ENUM_VALUE(WindingRule, ODD);
  SO_NODE_DEFINE_ENUM_VALUE(WindingRule, NONZERO);
  SO_NODE_DEFINE_ENUM_VALUE(WindingRule, POSITIVE);
  SO_NODE_DEFINE_ENUM_VALUE(WindingRule, NEGATIVE);
  SO_NODE_DEFINE_ENUM_VALUE(WindingRule, ABS_GEQ_TWO);
  SO_NODE_SET_SF_ENUM_TYPE(windingRule, WindingRule);
  SO_NODE_ADD_FIELD(windingRule, (ODD));
  SO_NODE_ADD_FIELD(coordinate, (nullptr));
  SO_NODE_ADD_FIELD(coordIndex, (-1));
}

IndexedTesselationFace::IndexedTesselationFace()  {
  constructor();
}

IndexedTesselationFace::IndexedTesselationFace(int numChilderen) : SoGroup(numChilderen) {
  constructor();
}

IndexedTesselationFace::~IndexedTesselationFace() = default;

void IndexedTesselationFace::write(SoWriteAction *action) {
  // the internal added children must not be write out

  // save all internal children
  int nr=getNumChildren();
  auto **child=new SoNode*[nr];
  for(int i=0; i<nr; i++) {
    child[i]=getChild(i);
    child[i]->ref();
  }
  // remove all internal children and write out without any internal children
  removeAllChildren();
  SoGroup::write(action);
  // restore all internal children
  for(int i=0; i<nr; i++) {
    addChild(child[i]);
    child[i]->unref();
  }
  delete[]child;
}

SbBool IndexedTesselationFace::readChildren(SoInput *in) {
  // if attributes are read, generate the children using gluTess

  Utils::initialize();

  int wr;
  switch(windingRule.getValue()) {
    case ODD: wr=GLU_TESS_WINDING_ODD; break;
    case NONZERO: wr=GLU_TESS_WINDING_NONZERO; break;
    case POSITIVE: wr=GLU_TESS_WINDING_POSITIVE; break;
    case NEGATIVE: wr=GLU_TESS_WINDING_NEGATIVE; break;
    default /*ABS_GEQ_TWO*/: wr=GLU_TESS_WINDING_ABS_GEQ_TWO; break;
  }
  gluTessProperty(Utils::tess(), GLU_TESS_WINDING_RULE, wr);
  gluTessBeginPolygon(Utils::tess(), this);
  bool contourOpen=false;
  for(int i=0; i<coordIndex.getNum(); i++) {
    if(!contourOpen && coordIndex[i]>=0) {
      gluTessBeginContour(Utils::tess());
      contourOpen=true;
    }
    if(coordIndex[i]>=0) {
      auto *v=(double*)(coordinate[coordIndex[i]].getValue());
      gluTessVertex(Utils::tess(), v, v);
    }
    if(coordIndex[i]<0 || i>=coordIndex.getNum()-1) {
      gluTessEndContour(Utils::tess());
      contourOpen=false;
    }
  }
  gluTessEndPolygon(Utils::tess());

  return true; // reading/generating of children sucessful
}

}
