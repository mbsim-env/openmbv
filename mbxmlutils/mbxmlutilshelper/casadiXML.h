#ifndef _MBXMLUTILS_CASADIXML_H_
#define _MBXMLUTILS_CASADIXML_H_

#include <mbxmlutilshelper/dom.h>
#include <map>
#include <limits>
#include <casadi/casadi.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <boost/lexical_cast.hpp>

namespace casadi {

const MBXMLUtils::NamespaceURI CASADI("http://www.mbsim-env.de/MBXMLUtils/CasADi");

inline xercesc::DOMElement *convertCasADiToXML_SXElem(const SXElem &s, std::map<SXNode*, int> &nodes, xercesc::DOMDocument *doc) {
  // add the node of s to the list of all nodes (creates a integer id for newly added nodes)
  std::string idStr;
  std::pair<std::map<SXNode*, int>::iterator, bool> ret=nodes.insert(std::make_pair(s.get(), nodes.size()));
  // if the node of s already exists in the list of all nodes write a reference to this node to XML
  if(ret.second==false) {
    xercesc::DOMElement *e=MBXMLUtils::D(doc)->createElement(CASADI%"SXElemRef");
    std::stringstream str;
    str<<ret.first->second;
    MBXMLUtils::E(e)->setAttribute("refid", str.str());
    return e;
  }
  // if the node of s does not exist in the list of all nodes set the id of this node
  std::stringstream str;
  str<<ret.first->second;
  idStr=str.str();

  // add s to XML dependent on the type of s
  xercesc::DOMElement *e;
  if(s.is_symbolic()) {
    e=MBXMLUtils::D(doc)->createElement(CASADI%"SymbolicSX");
    e->insertBefore(doc->createTextNode(MBXMLUtils::X()%s.name()), nullptr);
  }
  else if(s.is_zero())
    e=MBXMLUtils::D(doc)->createElement(CASADI%"ZeroSX");
  else if(s.is_one())
    e=MBXMLUtils::D(doc)->createElement(CASADI%"OneSX");
  else if(s.is_minus_one())
    e=MBXMLUtils::D(doc)->createElement(CASADI%"MinusOneSX");
  else if(s.isInf())
    e=MBXMLUtils::D(doc)->createElement(CASADI%"InfSX");
  else if(s.isMinusInf())
    e=MBXMLUtils::D(doc)->createElement(CASADI%"MinusInfSX");
  else if(s.isNan())
    e=MBXMLUtils::D(doc)->createElement(CASADI%"NanSX");
  else if(s.is_integer()) {
    e=MBXMLUtils::D(doc)->createElement(CASADI%"IntegerSX");
    std::stringstream str;
    str<<static_cast<int>(s);
    e->insertBefore(doc->createTextNode(MBXMLUtils::X()%str.str()), nullptr);
  }
  else if(s.is_constant()) {
    e=MBXMLUtils::D(doc)->createElement(CASADI%"RealtypeSX");
    std::stringstream str;
    str.precision(std::numeric_limits<double>::digits10+1);
    str<<static_cast<double>(s);
    e->insertBefore(doc->createTextNode(MBXMLUtils::X()%str.str()), nullptr);
  }
  else if(s.hasDep() && s.n_dep()==2) {
    e=MBXMLUtils::D(doc)->createElement(CASADI%"BinarySX");
    std::stringstream str;
    str<<s.op();
    MBXMLUtils::E(e)->setAttribute("op", str.str());
    e->insertBefore(convertCasADiToXML_SXElem(s.dep(0), nodes, doc), nullptr);
    e->insertBefore(convertCasADiToXML_SXElem(s.dep(1), nodes, doc), nullptr);
  }
  else if(s.hasDep() && s.n_dep()==1) {
    e=MBXMLUtils::D(doc)->createElement(CASADI%"UnarySX");
    std::stringstream str;
    str<<s.op();
    MBXMLUtils::E(e)->setAttribute("op", str.str());
    e->insertBefore(convertCasADiToXML_SXElem(s.dep(0), nodes, doc), nullptr);
  }
  else
    throw std::runtime_error("Unknown SXElem type in convertCasADiToXML_SXElem");

  // write also the id of a newly node to XML
  MBXMLUtils::E(e)->setAttribute("id", idStr);
  return e;
}

inline xercesc::DOMElement* convertCasADiToXML_SX(const SX &m, std::map<SXNode*, int> &nodes, xercesc::DOMDocument *doc) {
  // write each row of m to XML enclosed by a <row> element and each element in such rows to this element
  xercesc::DOMElement *e=MBXMLUtils::D(doc)->createElement(CASADI%"SX");
  for(int r=0; r<m.size1(); r++) {
    xercesc::DOMElement *row=MBXMLUtils::D(doc)->createElement(CASADI%"row");
    e->insertBefore(row, nullptr);
    for(int c=0; c<m.size2(); c++)
      row->insertBefore(convertCasADiToXML_SXElem(m(r, c).scalar(), nodes, doc), nullptr);
  }
  return e;
}

inline xercesc::DOMElement* convertCasADiToXML(const std::pair<std::vector<casadi::SX>, std::vector<casadi::SX>> &f, xercesc::DOMDocument *doc) {
  // write each input to XML enclosed by a <inputs> element
  std::map<SXNode*, int> nodes;
  xercesc::DOMElement *e=MBXMLUtils::D(doc)->createElement(CASADI%"Function");
  xercesc::DOMElement *input=MBXMLUtils::D(doc)->createElement(CASADI%"inputs");
  e->insertBefore(input, nullptr);
  for(const auto & i : f.first)
    input->insertBefore(convertCasADiToXML_SX(i, nodes, doc), nullptr);
  // write each output to XML enclosed by a <outputs> element
  xercesc::DOMElement *output=MBXMLUtils::D(doc)->createElement(CASADI%"outputs");
  e->insertBefore(output, nullptr);
  for(const auto & i : f.second)
    output->insertBefore(convertCasADiToXML_SX(i, nodes, doc), nullptr);

  return e;
}

inline SXElem createCasADiSXFromXML(xercesc::DOMElement *e, std::map<int, SXNode*> &nodes) {
  // creata an SXElem dependent on the type
  SXElem sxelement;
  if(MBXMLUtils::E(e)->getTagName()==CASADI%"BinarySX") {
    auto op = boost::lexical_cast<int>(MBXMLUtils::E(e)->getAttribute("op").c_str());
    xercesc::DOMElement *ee=e->getFirstElementChild();
    SXElem dep0(createCasADiSXFromXML(ee, nodes));
    ee=ee->getNextElementSibling();
    SXElem dep1(createCasADiSXFromXML(ee, nodes));
    sxelement=SXElem::binary(op, dep0, dep1);
  }
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"UnarySX") {
    auto op = boost::lexical_cast<int>(MBXMLUtils::E(e)->getAttribute("op").c_str());
    xercesc::DOMElement *ee=e->getFirstElementChild();
    SXElem dep=createCasADiSXFromXML(ee, nodes);
    sxelement=SXElem::unary(op, dep);
  }
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"SymbolicSX")
    sxelement=SXElem::sym(MBXMLUtils::X()%MBXMLUtils::E(e)->getFirstTextChild()->getData());
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"RealtypeSX")
    sxelement=boost::lexical_cast<double>(MBXMLUtils::X()%MBXMLUtils::E(e)->getFirstTextChild()->getData());
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"IntegerSX")
    sxelement=boost::lexical_cast<int>(MBXMLUtils::X()%MBXMLUtils::E(e)->getFirstTextChild()->getData());
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"ZeroSX")
    sxelement=casadi_limits<SXElem>::zero;
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"OneSX")
    sxelement=casadi_limits<SXElem>::one;
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"MinusOneSX")
    sxelement=casadi_limits<SXElem>::minus_one;
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"InfSX")
    sxelement=casadi_limits<SXElem>::inf;
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"MinusInfSX")
    sxelement=casadi_limits<SXElem>::minus_inf;
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"NanSX")
    sxelement=casadi_limits<SXElem>::nan;
  // reference elements must be handled specially: return the referenced node instead of creating a new one
  else if(MBXMLUtils::E(e)->getTagName()==CASADI%"SXElemRef") {
    auto refid = boost::lexical_cast<int>(MBXMLUtils::E(e)->getAttribute("refid").c_str());
    sxelement=SXElem::create(nodes[refid]);
    return sxelement;
  }
  else
    throw std::runtime_error("Unknown XML element named "+MBXMLUtils::X()%e->getTagName()+" in createCasADiSXFromXML");

  // insert a newly created SXElem (node) to the list of all nodes
  auto id = boost::lexical_cast<int>(MBXMLUtils::E(e)->getAttribute("id").c_str());
  nodes.insert(std::make_pair(id, sxelement.get()));
  return sxelement;
}

inline SX createCasADiSXMatrixFromXML(xercesc::DOMElement *e, std::map<int, SXNode*> &nodes) {
  // loop over all rows
  std::vector<std::vector<SXElem> > ret;
  xercesc::DOMElement *row=e->getFirstElementChild();
  while(row) {
    // loop over all elements in a row
    std::vector<SXElem> stdrow;
    xercesc::DOMElement *ele=row->getFirstElementChild();
    while(ele) {
      stdrow.push_back(createCasADiSXFromXML(ele, nodes));
      ele=ele->getNextElementSibling();
    }
    ret.push_back(stdrow);
    row=row->getNextElementSibling();
  }

  SX m=SX::zeros(ret.size(), ret[0].size());
  for(size_t c=0; c<ret[0].size(); ++c)
    for(size_t r=0; r<ret.size(); ++r)
      m(r,c)=ret[r][c];

  return m;
}

inline std::pair<std::vector<SX>, std::vector<SX>> createCasADiFunctionFromXML(xercesc::DOMElement *e) {
  // create a Function
  std::map<int, SXNode*> nodes;
  if(MBXMLUtils::E(e)->getTagName()==CASADI%"Function") {
    // get all inputs
    std::vector<SX> in;
    xercesc::DOMElement *input=e->getFirstElementChild();
    xercesc::DOMElement *inputEle=input->getFirstElementChild();
    while(inputEle) {
      in.push_back(createCasADiSXMatrixFromXML(inputEle, nodes));
      inputEle=inputEle->getNextElementSibling();
    }
    // get all outputs
    std::vector<SX> out;
    xercesc::DOMElement *output=input->getNextElementSibling();
    xercesc::DOMElement *outputEle=output->getFirstElementChild();
    while(outputEle) {
      out.push_back(createCasADiSXMatrixFromXML(outputEle, nodes));
      outputEle=outputEle->getNextElementSibling();
    }

    return make_pair(in, out);
  }
  else
    throw std::runtime_error("Unknown XML element named "+MBXMLUtils::X()%e->getTagName()+" in createCasADiFunctionFromXML.");
}

}

#endif
