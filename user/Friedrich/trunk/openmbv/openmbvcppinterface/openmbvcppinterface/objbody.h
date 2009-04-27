#ifndef _OPENMBV_OBJBODY_H_
#define _OPENMBV_OBJBODY_H_

#include <openmbvcppinterface/rigidbody.h>
#include <string>

namespace OpenMBV {

  /** A body defined by a Wavefront Obj file*/
  class ObjBody : public RigidBody {
    public:
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
      ObjBody();
      void setObjFileName(std::string objFileName_) { objFileName=objFileName_; }
      void setUseTextureFromMatLib(bool useTextureFromMatLib_) { useTextureFromMatLib=useTextureFromMatLib_; }
      void setUseMaterialFromMatLib(bool useMaterialFromMatLib_) { useMaterialFromMatLib=useMaterialFromMatLib_; }
      void setNormals(Normals normals_) { normals=normals_; }
      void setEpsVertex(double epsVertex_) { epsVertex=epsVertex_; }
      void setEpsNormal(double epsNormal_) { epsNormal=epsNormal_; }
      void setSmoothBarrier(double smoothBarrier_) { smoothBarrier=smoothBarrier_; }
      void setOutline(Outline outline_) { outline=outline_; }
    protected:
      std::string objFileName;
      bool useTextureFromMatLib, useMaterialFromMatLib;
      Normals normals;
      double epsVertex, epsNormal, smoothBarrier;
      Outline outline;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
  };

}

#endif
