#ifndef _OBJECT_H_
#define _OBJECT_H_

#include <string>
#include <H5Cpp.h>

namespace AMVis {

  class Group;

  class Object {
    friend class Group;
    protected:
      std::string name;
      Group* parent;
      virtual void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="")=0;
      virtual void createHDF5File()=0;
      H5::Group *hdf5Group;
    public:
      Object(const std::string& name_);
      ~Object();
      std::string getFullName();
  };

}

#endif
