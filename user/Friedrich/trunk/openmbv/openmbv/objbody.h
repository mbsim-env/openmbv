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
    static double eps;
    double smoothBarrier;
    void readMtlLib(const std::string& mtlFile, std::map<QString, SoMaterial*>& material);
    void readMapLib(const std::string& mtlFile, std::map<QString, SoTexture2*>& map_);

    // compares an Vertex like an alphanumeric string
    class SbVec3fHash {
      public:
        bool operator()(const SbVec3f& v1, const SbVec3f& v2) const;
    };

    // returns newvv with deleted duplicated vertices from vv;
    // also return newvi the a list of new indixies
    // complexibility: n*log(n)
    // v: IN vector of vertices
    // newv: OUT vector of new vertices
    // newvi: OUT vector of new indcies
    void combine(const SoMFVec3f& v, SoMFVec3f& newv, SoMFInt32& newvi);

    // substutute the indixies in fv with the new indixes newvi
    // complexibility: n
    // fvi: IN/OUT vector for face indicies
    // newvi: IN vector of new indcies
    void convertIndex(SoMFInt32& fvi, const SoMFInt32& newvi);

    // cal normals
    struct XXX {
      int ni1, ni2;
    };
    struct TwoIndex {
      int vi1, vi2;
    };
    class TwoIndexHash {
      public:
        bool operator()(const TwoIndex& l1, const TwoIndex& l2) const;
    };
    void computeNormals(const SoMFInt32& fvi, const SoMFVec3f &v, SoMFInt32& fni, SoMFVec3f& n, SoMFInt32& oli);
  public:
    ObjBody(TiXmlElement* element, H5::Group *h5Parent);
};

#endif
