#ifndef _AMVIS_BODY_H_
#define _AMVIS_BODY_H_

#include <string>
#include <amviscppinterface/object.h>

namespace AMVis {

  class Body : public Object {
    private:
      std::string getRelPathTo(Body* destBody);
    protected:
      Body* hdf5LinkBody;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      void terminate();
    public:
      Body();
      void setHDF5LinkTarget(Body* dest) { hdf5LinkBody=dest; }
      bool isHDF5Link() { return hdf5LinkBody!=0; }
  };

}

#endif
