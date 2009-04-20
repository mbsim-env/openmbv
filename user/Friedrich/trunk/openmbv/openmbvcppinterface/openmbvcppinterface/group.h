#ifndef _OPENMBV_GROUP_H_
#define _OPENMBV_GROUP_H_

#include <openmbvcppinterface/object.h>
#include <vector>
#include <H5Cpp.h>

namespace OpenMBV {

  class Group : public Object {
    friend class Body;
    protected:
      std::vector<Object*> object;
      bool separateFile;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
    public:
      Group();
      void addObject(Object* object);
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }
      void initialize();
      void terminate();
  };

}

#endif
