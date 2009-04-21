#ifndef _PATH_H_
#define _PATH_H_

#include "config.h"
#include "body.h"
#include "tinyxml.h"
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>

class Path : public Body {
  Q_OBJECT
  protected:
    virtual double update();
    H5::VectorSerie<double> *h5Data;
    SoCoordinate3 *coord;
    SoLineSet *line;
    int maxFrameRead;
  public:
    Path(TiXmlElement* element, H5::Group *h5Parent);
    virtual QString getInfo();
};

#endif
