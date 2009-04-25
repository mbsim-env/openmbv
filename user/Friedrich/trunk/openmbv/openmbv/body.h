#ifndef _BODY_H_
#define _BODY_H_

#include "config.h"
#include "object.h"
#include "tinyxml.h"
#include <Inventor/sensors/SoFieldSensor.h>
#include <H5Cpp.h>
#include <QtGui/QActionGroup>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <GL/glu.h>

class Body : public Object {
  Q_OBJECT
  private:
    enum DrawStyle { filled, lines, points };
    SoDrawStyle *drawStyle;
    static bool existFiles;
    static Body *timeUpdater; // the body who updates the time string in the scene window
  public:
    Body(TiXmlElement* element, H5::Group *h5Parent);
    static void frameSensorCB(void *data, SoSensor*);
    virtual QMenu* createMenu();
    virtual double update()=0; // return the current time
    SoSwitch *soOutLineSwitch;
    SoSeparator *soOutLineSep;
    QAction *outLine;
    QActionGroup *drawMethod;
    QAction *drawMethodPolygon, *drawMethodLine, *drawMethodPoint;
    void resetAnimRange(int numOfRows, double dt);
  public slots:
    void outLineSlot();
  protected slots:
    void drawMethodSlot(QAction* action);



  // FROM NOW ONLY CONVENIENCE FUNCTIONS FOLLOW !!!
  protected:
    // combine and calcaulate vertex/normal
    static double eps; // convenience
    // compares an Vertex like an alphanumeric string
    class SbVec3fHash { // convenience
      public:
        bool operator()(const SbVec3f& v1, const SbVec3f& v2) const;
    };
    // returns newvv with deleted duplicated vertices from vv;
    // also return newvi the a list of new indixies
    // complexibility: n*log(n)
    // v: IN vector of vertices
    // newv: OUT vector of new vertices
    // newvi: OUT vector of new indcies
    static void combine(const SoMFVec3f& v, SoMFVec3f& newv, SoMFInt32& newvi); // convenience
    // substutute the indixies in fv with the new indixes newvi
    // complexibility: n
    // fvi: IN/OUT vector for face indicies
    // newvi: IN vector of new indcies
    static void convertIndex(SoMFInt32& fvi, const SoMFInt32& newvi); // convenience
    // cal normals
    struct XXX { // convenience
      int ni1, ni2;
    };
    struct TwoIndex { // convenience
      int vi1, vi2;
    };
    class TwoIndexHash { // convenience
      public:
        bool operator()(const TwoIndex& l1, const TwoIndex& l2) const;
    };
    static void computeNormals(const SoMFInt32& fvi, const SoMFVec3f &v, SoMFInt32& fni, SoMFVec3f& n, SoMFInt32& oli, double smoothBarrier); // convenience

    // tess // convenience
    static GLUtesselator *tess; // convenience
    static GLenum tessType; // convenience
    static int tessNumVertices; // convenience
    static SoTriangleStripSet *tessTriangleStrip; // convenience
    static SoIndexedFaceSet *tessTriangleFan; // convenience
    static SoCoordinate3 *tessCoord; // convenience
    static bool tessCBInit; // convenience
    static void tessBeginCB(GLenum type, void *data); // convenience
    static void tessVertexCB(GLdouble *vertex); // convenience
    static void tessEndCB(void); // convenience

    static std::vector<double> toVector(std::string str); // convenience
    static std::vector<std::vector<double> > toMatrix(std::string str); // convenience
    static SoSeparator* soFrame(double size, double offset) { SoScale *scale; return soFrame(size, offset, scale); } // convenience
    static SoSeparator* soFrame(double size, double offset, SoScale *&scale); // convenience
};

#endif
