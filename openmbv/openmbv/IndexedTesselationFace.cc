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

  // on change on EVERY field call the changedCB method
  sensor=new SoNodeSensor(changedCB, this);
  sensor->attach(this);
  sensor->setPriority(100);
}

IndexedTesselationFace::IndexedTesselationFace() : SoGroup() {
  constructor();
}

IndexedTesselationFace::IndexedTesselationFace(int numChilderen) : SoGroup(numChilderen) {
  constructor();
}

IndexedTesselationFace::~IndexedTesselationFace() {
}

void IndexedTesselationFace::changedCB(void *data, SoSensor*) {
  assert(Body::tessCBInit);
  IndexedTesselationFace *me=(IndexedTesselationFace*)data;

  me->sensor->detach();

  me->removeAllChildren();

  int wr;
  switch(me->windingRule.getValue()) {
    case ODD: wr=GLU_TESS_WINDING_ODD; break;
    case NONZERO: wr=GLU_TESS_WINDING_NONZERO; break;
    case POSITIVE: wr=GLU_TESS_WINDING_POSITIVE; break;
    case NEGATIVE: wr=GLU_TESS_WINDING_NEGATIVE; break;
    case ABS_GEQ_TWO: wr=GLU_TESS_WINDING_ABS_GEQ_TWO; break;
  }
  gluTessProperty(Body::tess, GLU_TESS_WINDING_RULE, wr);
  gluTessBeginPolygon(Body::tess, me);
  bool contourOpen=false;
  for(int i=0; i<me->coordIndex.getNum(); i++) {
    if(contourOpen==false && me->coordIndex[i]>=0) {
      gluTessBeginContour(Body::tess);
      contourOpen=true;
    }
    if(me->coordIndex[i]>=0) {
      double *v=(double*)(me->coordinate[me->coordIndex[i]].getValue());
      gluTessVertex(Body::tess, v, v);
    }
    if(me->coordIndex[i]<0 || i>=me->coordIndex.getNum()-1) {
      gluTessEndContour(Body::tess);
      contourOpen=false;
    }
  }
  gluTessEndPolygon(Body::tess);

  me->sensor->attach(me);
}
