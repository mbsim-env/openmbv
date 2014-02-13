#ifndef _MBXMLUTILS_CASADIXML_H_
#define _MBXMLUTILS_CASADIXML_H_

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <casadi/symbolic/sx/sx.hpp>
#include <casadi/symbolic/fx/sx_function.hpp>
#include <casadi/symbolic/fx/sx_function.hpp>
#include <mbxmlutilshelper/dom.h>
#include <set>
#include <memory>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMImplementation.hpp>


namespace CasADi {

using namespace std;
using namespace xercesc;
using namespace boost;
using namespace MBXMLUtils;

extern NamespaceURI CASADI;

inline DOMElement *convertCasADiToXML(const SX &s, map<SXNode*, int> &nodes, DOMDocument *doc) {
  // add the node of s to the list of all nodes (creates a integer id for newly added nodes)
  string idStr;
  pair<map<SXNode*, int>::iterator, bool> ret=nodes.insert(make_pair(s.get(), nodes.size()));
  // if the node of s already exists in the list of all nodes write a reference to this node to XML
  if(ret.second==false) {
    DOMElement *e=D(doc)->createElement(CASADI%"reference");
    stringstream str;
    str<<ret.first->second;
    E(e)->setAttribute("refid", str.str());
    return e;
  }
  // if the node of s does not exist in the list of all nodes set the id of this node
  else {
    stringstream str;
    str<<ret.first->second;
    idStr=str.str();
  }

  // add s to XML dependent on the type of s
  DOMElement *e;
  if(s.isSymbolic()) {
    e=D(doc)->createElement(CASADI%"SymbolicSX");
    e->insertBefore(doc->createTextNode(X()%s.getName()), NULL);
  }
  else if(s.isZero())
    e=D(doc)->createElement(CASADI%"ZeroSX");
  else if(s.isOne())
    e=D(doc)->createElement(CASADI%"OneSX");
  else if(s.isMinusOne())
    e=D(doc)->createElement(CASADI%"MinusOneSX");
  else if(s.isInf())
    e=D(doc)->createElement(CASADI%"InfSX");
  else if(s.isMinusInf())
    e=D(doc)->createElement(CASADI%"MinusInfSX");
  else if(s.isNan())
    e=D(doc)->createElement(CASADI%"NanSX");
  else if(s.isInteger()) {
    e=D(doc)->createElement(CASADI%"IntegerSX");
    stringstream str;
    str<<s.getIntValue();
    e->insertBefore(doc->createTextNode(X()%str.str()), NULL);
  }
  else if(s.isConstant()) {
    e=D(doc)->createElement(CASADI%"RealtypeSX");
    stringstream str;
    str.precision(18);
    str<<s.getValue();
    e->insertBefore(doc->createTextNode(X()%str.str()), NULL);
  }
  else if(s.hasDep() && s.getNdeps()==2) {
    e=D(doc)->createElement(CASADI%"BinarySX");
    stringstream str;
    str<<s.getOp();
    E(e)->setAttribute("op", str.str());
    e->insertBefore(convertCasADiToXML(s.getDep(0), nodes, doc), NULL);
    e->insertBefore(convertCasADiToXML(s.getDep(1), nodes, doc), NULL);
  }
  else if(s.hasDep() && s.getNdeps()==1) {
    e=D(doc)->createElement(CASADI%"UnarySX");
    stringstream str;
    str<<s.getOp();
    E(e)->setAttribute("op", str.str());
    e->insertBefore(convertCasADiToXML(s.getDep(0), nodes, doc), NULL);
  }
  else
    throw runtime_error("Unknown CasADi::SX type in convertCasADiToXML");

  // write also the id of a newly node to XML
  E(e)->setAttribute("id", idStr);
  return e;
}

inline DOMElement* convertCasADiToXML(const SXMatrix &m, map<SXNode*, int> &nodes, DOMDocument *doc) {
  // if it is a scalar print it as a scalar
  if(m.size1()==1 && m.size2()==1)
    return convertCasADiToXML(m.elem(0, 0), nodes, doc);
  // write each matrixRow of m to XML enclosed by a <matrixRow> element and each element in such rows 
  // to this element
  DOMElement *e=D(doc)->createElement(CASADI%"SXMatrix");
  if(m.size1()==1) E(e)->setAttribute("rowVector", "true");
  if(m.size2()==1) E(e)->setAttribute("columnVector", "true");
  for(int r=0; r<m.size1(); r++) {
    DOMElement *matrixRow;
    if(m.size1()==1 || m.size2()==1)
      matrixRow=e;
    else {
      matrixRow=D(doc)->createElement(CASADI%"matrixRow");
      e->insertBefore(matrixRow, NULL);
    }
    for(int c=0; c<m.size2(); c++)
      matrixRow->insertBefore(convertCasADiToXML(m.elem(r, c), nodes, doc), NULL);
  }
  return e;
}

inline DOMElement* convertCasADiToXML(const SXFunction &f, DOMDocument *doc) {
  // write each input of f to XML enclosed by a <inputs> element
  map<SXNode*, int> nodes;
  DOMElement *e=D(doc)->createElement(CASADI%"SXFunction");
  const vector<SXMatrix> &in=f.inputExpr();
  DOMElement *input=D(doc)->createElement(CASADI%"inputs");
  e->insertBefore(input, NULL);
  for(size_t i=0; i<in.size(); i++)
    input->insertBefore(convertCasADiToXML(in[i], nodes, doc), NULL);
  // write each output of f to XML enclosed by a <outputs> element
  const vector<SXMatrix> &out=f.outputExpr();
  DOMElement *output=D(doc)->createElement(CASADI%"outputs");
  e->insertBefore(output, NULL);
  for(size_t i=0; i<out.size(); i++)
    output->insertBefore(convertCasADiToXML(out[i], nodes, doc), NULL);

  return e;
}

//inline CasADi::SX createCasADiSXFromXML(MBXMLUtils::TiXmlElement *e, std::map<int, CasADi::SXNode*> &nodes) {
//  // creata an SX dependent on the type
//  CasADi::SX sx;
//  if(e->ValueStr()==MBXMLUTILSCASADINS"BinarySX") {
//    int op;
//    e->QueryIntAttribute("op", &op);
//    MBXMLUtils::TiXmlElement *ee=e->FirstChildElement();
//    CasADi::SX dep0=createCasADiSXFromXML(ee, nodes);
//    ee=ee->NextSiblingElement();
//    CasADi::SX dep1=createCasADiSXFromXML(ee, nodes);
//    sx=CasADi::SX::binary(op, dep0, dep1);
//  }
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"UnarySX") {
//    int op;
//    e->QueryIntAttribute("op", &op);
//    MBXMLUtils::TiXmlElement *ee=e->FirstChildElement();
//    CasADi::SX dep=createCasADiSXFromXML(ee, nodes);
//    sx=CasADi::SX::unary(op, dep);
//  }
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"SymbolicSX") {
//    MBXMLUtils::TiXmlText *ee=e->FirstChildText();
//    sx=CasADi::SX(ee->ValueStr());
//  }
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"RealtypeSX") {
//    MBXMLUtils::TiXmlText *ee=e->FirstChildText();
//    std::stringstream str(ee->ValueStr());
//    double value;
//    str>>value;
//    sx=value;
//  }
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"IntegerSX") {
//    MBXMLUtils::TiXmlText *ee=e->FirstChildText();
//    std::stringstream str(ee->ValueStr());
//    int value;
//    str>>value;
//    sx=value;
//  }
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"ZeroSX")
//    sx=CasADi::casadi_limits<CasADi::SX>::zero;
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"OneSX")
//    sx=CasADi::casadi_limits<CasADi::SX>::one;
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"MinusOneSX")
//    sx=CasADi::casadi_limits<CasADi::SX>::minus_one;
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"InfSX")
//    sx=CasADi::casadi_limits<CasADi::SX>::inf;
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"MinusInfSX")
//    sx=CasADi::casadi_limits<CasADi::SX>::minus_inf;
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"NanSX")
//    sx=CasADi::casadi_limits<CasADi::SX>::nan;
//  // reference elements must be handled specially: return the referenced node instead of creating a new one
//  else if(e->ValueStr()==MBXMLUTILSCASADINS"reference") {
//    int refid;
//    e->QueryIntAttribute("refid", &refid);
//    sx=CasADi::SX(nodes[refid], false);
//    return sx;
//  }
//  else
//    throw std::runtime_error("Unknown XML element named "+e->ValueStr()+" in createCasADiSXFromXML");
//
//  // insert a newly created SX (node) to the list of all nodes
//  int id;
//  e->QueryIntAttribute("id", &id);
//  nodes.insert(std::make_pair(id, sx.get()));
//  return sx;
//}
//
//inline CasADi::SXMatrix createCasADiSXMatrixFromXML(MBXMLUtils::TiXmlElement *e, std::map<int, CasADi::SXNode*> &nodes) {
//  // create a SXMatrix
//  if(e->ValueStr()==MBXMLUTILSCASADINS"SXMatrix") {
//    // loop over all rows
//    std::vector<std::vector<CasADi::SX> > ret;
//    MBXMLUtils::TiXmlElement *matrixRow=e->FirstChildElement();
//    while(matrixRow) {
//      // loop over all elements in a matrixRow
//      std::vector<CasADi::SX> stdrow;
//      MBXMLUtils::TiXmlElement *matrixEle;
//      if((e->Attribute("rowVector") && e->Attribute("rowVector")==std::string("true")) ||
//         (e->Attribute("columnVector") && e->Attribute("columnVector")==std::string("true")))
//        matrixEle=matrixRow;
//      else
//        matrixEle=matrixRow->FirstChildElement();
//      while(matrixEle) {
//        stdrow.push_back(createCasADiSXFromXML(matrixEle, nodes));
//        matrixEle=matrixEle->NextSiblingElement();
//      }
//      if((e->Attribute("rowVector") && e->Attribute("rowVector")==std::string("true")) ||
//         (e->Attribute("columnVector") && e->Attribute("columnVector")==std::string("true")))
//        matrixRow=matrixEle;
//      else
//        matrixRow=matrixRow->NextSiblingElement();
//      ret.push_back(stdrow);
//    }
//    if(e->Attribute("columnVector") && e->Attribute("columnVector")==std::string("true"))
//      return CasADi::SXMatrix(ret).trans();
//    else
//      return CasADi::SXMatrix(ret);
//  }
//  // if it is a scalar create a SX
//  else
//    return createCasADiSXFromXML(e, nodes);
//}
//
//inline CasADi::SXFunction createCasADiSXFunctionFromXML(MBXMLUtils::TiXmlElement *e) {
//  // create a SXFunction
//  std::map<int, CasADi::SXNode*> nodes;
//  if(e->ValueStr()==MBXMLUTILSCASADINS"SXFunction") {
//    // get all inputs
//    std::vector<CasADi::SXMatrix> in;
//    MBXMLUtils::TiXmlElement *input=e->FirstChildElement();
//    MBXMLUtils::TiXmlElement *inputEle=input->FirstChildElement();
//    while(inputEle) {
//      in.push_back(createCasADiSXMatrixFromXML(inputEle, nodes));
//      inputEle=inputEle->NextSiblingElement();
//    }
//    // get all outputs
//    std::vector<CasADi::SXMatrix> out;
//    MBXMLUtils::TiXmlElement *output=input->NextSiblingElement();
//    MBXMLUtils::TiXmlElement *outputEle=output->FirstChildElement();
//    while(outputEle) {
//      out.push_back(createCasADiSXMatrixFromXML(outputEle, nodes));
//      outputEle=outputEle->NextSiblingElement();
//    }
//
//    return CasADi::SXFunction(in, out);
//  }
//  else
//    throw std::runtime_error("Unknown XML element named "+e->ValueStr()+" in createCasADiSXFunctionFromXML.");
//}

}

#endif
