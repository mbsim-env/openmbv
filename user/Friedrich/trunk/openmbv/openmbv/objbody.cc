#include "config.h"
#include "objbody.h"
#include <QFile>
#include <QTextStream>
#include <tinyxml/tinynamespace.h>
#include <Inventor/nodes/SoShapeHints.h>

using namespace std;

double ObjBody::eps;

ObjBody::ObjBody(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
  iconFile=":/objbody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"objFileName");
  string fileName=e->GetText();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=fixPath(e->GetDocument()->ValueStr(), fileName);
  e=e->NextSiblingElement();
  bool textureFromFile=e->GetText()=="true"?true:false;
  e=e->NextSiblingElement();
  bool materialFromFile=e->GetText()=="true"?true:false;
  e=e->NextSiblingElement();
  string normals_=e->GetText();
  Normals normals;
  if(normals_=="fromObjFile") normals=fromObjFile;
  else if(normals_=="flat") normals=flat;
  else if(normals_=="smooth") normals=smooth;
  else normals=smoothIfLessBarrier;
  e=e->NextSiblingElement();
  double epsVertex=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double epsNormal=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  smoothBarrier=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  string outline_=e->GetText();
  Outline outline;
  if(outline_=="none") outline=none;
  else if(outline_=="calculate") outline=calculate;
  else outline=fromFile;

  // create so
  // read obj
  QFile objFile(fileName.c_str());
  objFile.open(QIODevice::ReadOnly | QIODevice::Text);
  map<QString, ObjGroup> group;
  ObjGroup* curGroup=&group["OpenMBVDefaultGroup"];
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp groupRE("[ \t]*g[ \t].*");
  QRegExp vRE("[ \t]*v[ \t].*");
  QRegExp vnRE("[ \t]*vn[ \t].*");
  QRegExp fRE("[ \t]*f[ \t].*");
  QRegExp fSplitRE("^([0-9]+)(?:/([0-9]*)(?:/([0-9]*))?)?$");
  while(!objFile.atEnd()) {
    QByteArray line=objFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(groupRE.exactMatch(line)) { // set current group
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      curGroup=&group[dummyStr];
      continue;
    }
    if(vRE.exactMatch(line)) { // read v (write to current group)
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      curGroup->v->point.set1Value(curGroup->v->point.getNum(), x,y,z);
      continue;
    }
    if(vnRE.exactMatch(line)) { // read vn (write to current group)
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      curGroup->n->vector.set1Value(curGroup->n->vector.getNum(), x,y,z);
      continue;
    }
    if(fRE.exactMatch(line)) { // read f (write to current group)
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi1=fSplitRE.cap(1).toInt()-1, ni1=fSplitRE.cap(3).toInt()-1;
      if(ni1<0) curGroup->hasNormal=false;
      stream>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi2=fSplitRE.cap(1).toInt()-1, ni2=fSplitRE.cap(3).toInt()-1;
      if(ni2<0) curGroup->hasNormal=false;
      while(!stream.atEnd()) {
        stream>>dummyStr;
        if(dummyStr=="") break;
        fSplitRE.indexIn(dummyStr);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi1);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni1);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        vi2=fSplitRE.cap(1).toInt()-1; ni2=fSplitRE.cap(3).toInt()-1;
        if(ni2<0) curGroup->hasNormal=false;
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), -1);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), -1);
      }
      continue;
    }
  }
  // create so
  map<QString, ObjGroup>::iterator i;
  for(i=group.begin(); i!=group.end(); i++) {
    if(epsVertex>0) { // combine vertex
      SoMFVec3f newvv;
      SoMFInt32 newvi;
      eps=epsVertex;
      combine(i->second.v->point, newvv, newvi);
      convertIndex(i->second.f->coordIndex, newvi);
      i->second.v->point.copyFrom(newvv);
    }
    soSep->addChild(i->second.v);
    if(normals==flat) {
      SoNormalBinding *nb=new SoNormalBinding;
      nb->value.setValue(SoNormalBinding::PER_FACE);
    }
    else if(normals==smooth || normals==smoothIfLessBarrier) {
      if(normals==smooth) smoothBarrier=M_PI;
      SoMFInt32 fni;
      SoMFVec3f n;
      SoMFInt32 oli;
      computeNormals(i->second.f->coordIndex, i->second.v->point, fni, n, oli);
      i->second.f->normalIndex.copyFrom(fni);
      i->second.n->vector.copyFrom(n);
      soSep->addChild(i->second.n);
      i->second.l->coordIndex.copyFrom(oli);
    }
    else {
      if(i->second.hasNormal || normals==fromObjFile) // if hasNormal draw with normals
        soSep->addChild(i->second.n);
      else { // it !hasNormal draw with automatic normals by Inventor and not backface culled and two-sided lighting
        SoShapeHints *sh=new SoShapeHints;
        soSep->addChild(sh);
        sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
        sh->shapeType.setValue(SoShapeHints::UNKNOWN_SHAPE_TYPE);
      }
    }
    if(epsNormal>0) { // combine normal
      SoMFVec3f newnv;
      SoMFInt32 newni;
      eps=epsNormal;
      combine(i->second.n->vector, newnv, newni);
      convertIndex(i->second.f->normalIndex, newni);
      i->second.n->vector.copyFrom(newnv);
    }
    soSep->addChild(i->second.f);
  }

  // outline
  soSep->addChild(soOutLineSwitch);
  for(i=group.begin(); i!=group.end(); i++) {
    if(i->second.l->coordIndex.getNum()>0) soOutLineSep->addChild(i->second.v);
    if(i->second.l->coordIndex.getNum()>0) soOutLineSep->addChild(i->second.l);
  }
}

ObjBody::ObjGroup::ObjGroup() {
  v=new SoCoordinate3; v->ref(); v->point.deleteValues(0);
  n=new SoNormal; n->ref(); n->vector.deleteValues(0);
  f=new SoIndexedFaceSet; f->ref(); f->coordIndex.deleteValues(0); f->normalIndex.deleteValues(0);
  l=new SoIndexedLineSet; l->ref(); l->coordIndex.deleteValues(0);
  hasNormal=true;
}

bool ObjBody::SbVec3fHash::operator()(const SbVec3f& v1, const SbVec3f& v2) const {
  float x1,y1,z1,x2,y2,z2;
  v1.getValue(x1,y1,z1);
  v2.getValue(x2,y2,z2);
  if(x1<x2-eps) return true;
  else if(x1>x2+eps) return false;
  else
    if(y1<y2-eps) return true;
    else if(y1>y2+eps) return false;
    else
      if(z1<z2-eps) return true;
      else if(z1>z2+eps) return false;
      else return false;
};

void ObjBody::combine(const SoMFVec3f& v, SoMFVec3f& newv, SoMFInt32& newvi) {
  map<SbVec3f, int, SbVec3fHash> hash;
  for(int i=0; i<v.getNum(); i++) {
    int &r=hash[*(v.getValues(i))]; // add vertex to a hash map
    // if vertex not exist in hash map copy it to newvv,
    // set corospondenting newvi,
    // and set hash map value to the index of this vertex
    if(r==0) {
      newv.set1Value(newv.getNum(), *v.getValues(i));
      newvi.set1Value(i, newv.getNum()-1);
      r=newv.getNum()-1+1;
    }
    // if vertix exist in hash map,
    // set corrospondenting newvi to the value in the hash map
    else
      newvi.set1Value(i, r-1);
  }
}

void ObjBody::convertIndex(SoMFInt32& fvi, const SoMFInt32& newvi) {
  for(int i=0; i<fvi.getNum(); i++)
    if(fvi[i]>=0) fvi.set1Value(i, newvi[fvi[i]]);
}

// complexibility: ?
bool ObjBody::TwoIndexHash::operator()(const TwoIndex& l1, const TwoIndex& l2) const {
  if(l1.vi1<l2.vi1) return true;
  else if(l1.vi1>l2.vi1) return false;
  else
    if(l1.vi2<l2.vi2) return true;
    else if(l1.vi2>l2.vi2) return false;
    else return false;
}

void ObjBody::computeNormals(const SoMFInt32& fvi, const SoMFVec3f &v, SoMFInt32& fni, SoMFVec3f& n, SoMFInt32& oli) {
  map<TwoIndex, vector<XXX>, TwoIndexHash> lni; // line normal index = index of normal of start/end point
  map<int, vector<int> > vni;

  for(int i=0; i<fvi.getNum(); i+=4) {
    // set face normals fn
    SbVec3f fn, ft, fb;
    ft=v[fvi[i+1]]-v[fvi[i+0]];
    fb=v[fvi[i+2]]-v[fvi[i+0]];
    fn=ft.cross(fb);
    fni.set1Value(i+0, i+0);
    fni.set1Value(i+1, i+1);
    fni.set1Value(i+2, i+2);
    fni.set1Value(i+3, -1);
    n.set1Value(i+0, fn);
    n.set1Value(i+1, fn);
    n.set1Value(i+2, fn);

    // store all face indexies of each line in a map
    TwoIndex l;
    XXX xxx;
    l.vi1=fvi[i+0]; l.vi2=fvi[i+1];
    xxx.ni1=i+0; xxx.ni2=i+1;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);
    l.vi1=fvi[i+1]; l.vi2=fvi[i+2];
    xxx.ni1=i+1; xxx.ni2=i+2;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);
    l.vi1=fvi[i+2]; l.vi2=fvi[i+0];
    xxx.ni1=i+2; xxx.ni2=i+0;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);

    // vni
    vni[fvi[i+0]].push_back(i+0);
    vni[fvi[i+1]].push_back(i+1);
    vni[fvi[i+2]].push_back(i+2);
  }
  map<TwoIndex, vector<XXX>, TwoIndexHash>::iterator i;
  int ni1, ni2;
  SbVec3f nNew;
  for(i=lni.begin(); i!=lni.end(); i++) {
    if(i->second.size()!=2) continue;
    bool smooth=false;
    ni1=i->second[0].ni1; ni2=i->second[1].ni1;
    if(acos(n[fni[ni1]].dot(n[fni[ni2]])/n[fni[ni1]].length()/n[fni[ni2]].length())<smoothBarrier) {
      smooth=true;
      nNew=n[fni[ni1]]+n[fni[ni2]];
      n.set1Value(fni[ni1], nNew); n.set1Value(fni[ni2], nNew);
      vector<int> vvv=vni[i->first.vi1];
      for(int k=0; k<vvv.size(); k++)
        if(fni[vvv[k]]==fni[ni2]) fni.set1Value(vvv[k], fni[ni1]);
    }
    ni1=i->second[0].ni2; ni2=i->second[1].ni2;
    if(acos(n[fni[ni1]].dot(n[fni[ni2]])/n[fni[ni1]].length()/n[fni[ni2]].length())<smoothBarrier) {
      smooth=true;
      nNew=n[fni[ni1]]+n[fni[ni2]];
      n.set1Value(fni[ni1], nNew); n.set1Value(fni[ni2] ,nNew);
      vector<int> vvv=vni[i->first.vi1];
      for(int k=0; k<vvv.size(); k++)
        if(fni[vvv[k]]==fni[ni2]) fni.set1Value(vvv[k], fni[ni1]);
    }
    if(!smooth) {
      oli.set1Value(oli.getNum(), fvi[i->second[0].ni1]);
      oli.set1Value(oli.getNum(), fvi[i->second[0].ni2]);
      oli.set1Value(oli.getNum(), -1);
    }
  }
}
