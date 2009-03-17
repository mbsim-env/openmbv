#ifndef _BODY_H_
#define _BODY_H_

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
    public:
      Body(const std::string& name_);
      void setHDF5Link(Body* dest) {
        hdf5LinkBody=dest;
      }
  };

}

#endif
