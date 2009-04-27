#ifndef _OPENMBV_BODY_H_
#define _OPENMBV_BODY_H_

#include <string>
#include <openmbvcppinterface/object.h>

namespace OpenMBV {

  /** Abstract base class for all bodies */
  class Body : public Object {
    private:
      std::string getRelPathTo(Body* destBody);
    protected:
      Body* hdf5LinkBody;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      void terminate();
    public:
      /** Default constructor */
      Body();

      /** Link this body with dest in the HDF5 file */
      void setHDF5LinkTarget(Body* dest) { hdf5LinkBody=dest; }

      /** Returns if this body is linked to another */
      bool isHDF5Link() { return hdf5LinkBody!=0; }
  };

}

#endif
