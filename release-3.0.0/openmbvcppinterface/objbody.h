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

      /** Default constructor */
      ObjBody();

      /** The obj file to read */
      void setObjFileName(std::string objFileName_) { objFileName=objFileName_; }

      /** Use the texture information in the material file if true */
      void setUseTextureFromMatLib(bool useTextureFromMatLib_) { useTextureFromMatLib=useTextureFromMatLib_; }

      /** Use the materlial information in the material file if true */
      void setUseMaterialFromMatLib(bool useMaterialFromMatLib_) { useMaterialFromMatLib=useMaterialFromMatLib_; }

      /** Set how normals should be generated.
       * Use "flat" for flat rendering of all faces.
       * Use "smooth" for smooth rendering of all faces.
       * Use "fromObjFile" to use the normals specified in the obj file.
       * Use "smoothIfLessBarrier" to calcualte smooth normals only if the angle between
       * two faces is smaller then smoothBarrier.
       * To use smoothIfLessBarrier you must likely set the eps value for vertices.
       */
      void setNormals(Normals normals_) { normals=normals_; }

      /** set the barrier below two vertices should be considered as equal.
       * This value is only used if outline calculation is enabled.
       */
      void setEpsVertex(double epsVertex_) { epsVertex=epsVertex_; }

      /** Set the barrier below two normals should be considered as equal.
       * This value is only used if outline calculation is enabled.
       */
      void setEpsNormal(double epsNormal_) { epsNormal=epsNormal_; }

      /** Set the barrier below smooth normals are calculated.
       * Normals must be set to "smoothIfLessBarrier"
       */
      void setSmoothBarrier(double smoothBarrier_) { smoothBarrier=smoothBarrier_; }

      /** Set how outlines should be generated.
       * Use "none" for no outline.
       * Use "calcualte" to calculate outline internally.
       * Use "fromFile" to read (out) lines from the obj file.
       * To use "calculate" the Normals must be set to "smoothIfLessBarrier" and
       */
      void setOutline(Outline outline_) { outline=outline_; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
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
