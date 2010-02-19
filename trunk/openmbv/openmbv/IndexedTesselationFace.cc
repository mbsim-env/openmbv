#include "IndexedTesselationFace.h"
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <GL/glu.h>
#include "body.h"

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
  SO_NODE_ADD_FIELD(coordinate, (NULL));
  SO_NODE_ADD_FIELD(coordIndex, (-1));
}

IndexedTesselationFace::IndexedTesselationFace() : SoGroup() {
  constructor();
}

IndexedTesselationFace::IndexedTesselationFace(int numChilderen) : SoGroup(numChilderen) {
  constructor();
}

IndexedTesselationFace::~IndexedTesselationFace() {
}

SbBool IndexedTesselationFace::readChildren(SoInput *in) {
  // if attributes are read, generate the children using gluTess

  assert(Body::tessCBInit);

  int wr;
  switch(windingRule.getValue()) {
    case ODD: wr=GLU_TESS_WINDING_ODD; break;
    case NONZERO: wr=GLU_TESS_WINDING_NONZERO; break;
    case POSITIVE: wr=GLU_TESS_WINDING_POSITIVE; break;
    case NEGATIVE: wr=GLU_TESS_WINDING_NEGATIVE; break;
    default /*ABS_GEQ_TWO*/: wr=GLU_TESS_WINDING_ABS_GEQ_TWO; break;
  }
  gluTessProperty(Body::tess, GLU_TESS_WINDING_RULE, wr);
  gluTessBeginPolygon(Body::tess, this);
  bool contourOpen=false;
  for(int i=0; i<coordIndex.getNum(); i++) {
    if(contourOpen==false && coordIndex[i]>=0) {
      gluTessBeginContour(Body::tess);
      contourOpen=true;
    }
    if(coordIndex[i]>=0) {
      double *v=(double*)(coordinate[coordIndex[i]].getValue());
      gluTessVertex(Body::tess, v, v);
    }
    if(coordIndex[i]<0 || i>=coordIndex.getNum()-1) {
      gluTessEndContour(Body::tess);
      contourOpen=false;
    }
  }
  gluTessEndPolygon(Body::tess);

  return true; // reading/generating of children sucessful
}
