#include <string>
#include <list>
#include <iostream>
#include <octeval.h>
#include <mbxmlutilstinyxml/getinstallpath.h>
#include <libxml/xmlschemas.h>
#include <libxml/xinclude.h>
#ifdef HAVE_UNORDERED_SET
#  include <unordered_set>
#else
#  include <set>
#  define unordered_set set
#endif
#include "mbxmlutilstinyxml/tinynamespace.h"

using namespace std;
using namespace MBXMLUtils;
namespace bfs=boost::filesystem;

bfs::path SCHEMADIR;

void addFilesInDir(vector<bfs::path> &dependencies, const bfs::path &dir, const bfs::path &ext) {
  for(bfs::directory_iterator it=bfs::directory_iterator(dir); it!=bfs::directory_iterator(); it++)
    if(it->path().extension()==ext)
      dependencies.push_back(it->path());
}

void warningCallback(void *ctx, const char *msg, ...) {
  // Schemas may be loaded from e.g aa/bb.xsd and the same schema from aa/../aa/bb.xsd
  // This is misleadingly interpreted on Windows as different schemas.
  // Hence we disable the corresponding warning (if not MBXMLUTILS_SCHEMAWARNINGS is set)
  static int printWarning=-1;
  if(printWarning==-1 && getenv("MBXMLUTILS_SCHEMAWARNINGS")!=NULL) printWarning=1; else printWarning=0;
  if(printWarning) {
    va_list ap;
    va_start(ap, msg);
    vfprintf(stderr, msg, ap);
    va_end(ap);
  }
}
// validate file using schema (currently by libxml)
void validate(const bfs::path &schema, const bfs::path &file) {
  cout<<"Parse and validate "<<file<<endl;

  // cache alread validated files
  static unordered_set<bfs::path> validatedCache;
  pair<unordered_set<bfs::path>::iterator, bool> ins2=validatedCache.insert(file);
  if(!ins2.second)
    return;

  boost::shared_ptr<xmlDoc> doc;
  doc.reset(xmlReadFile(file.c_str(), NULL, XML_PARSE_XINCLUDE | XML_PARSE_NOXINCNODE | XML_PARSE_NOBASEFIX),
            &xmlFreeDoc);
  if(!doc.get()) throw runtime_error("Internal error: can not load schema file.");
  if(xmlXIncludeProcess(doc.get())<0) throw runtime_error("Internal error: can not xinclude schema file.");

  // cache compiled schema
  static unordered_map<bfs::path, boost::shared_ptr<xmlSchemaValidCtxt> > compiledSchemaCache;
  pair<unordered_map<bfs::path, boost::shared_ptr<xmlSchemaValidCtxt> >::iterator, bool> ins=compiledSchemaCache.insert(
    make_pair(schema, boost::shared_ptr<xmlSchemaValidCtxt>()));
  if(ins.second) {
    boost::shared_ptr<xmlSchemaParserCtxt> schemaCtxt(xmlSchemaNewParserCtxt(schema.generic_string().c_str()),
                                                      &xmlSchemaFreeParserCtxt);
    xmlSchemaSetParserErrors(schemaCtxt.get(), NULL, &warningCallback, NULL); // redirect warning messages
    ins.first->second=boost::shared_ptr<xmlSchemaValidCtxt>(xmlSchemaNewValidCtxt(xmlSchemaParse(schemaCtxt.get())),
                                                            &xmlSchemaFreeValidCtxt);
  }

  int ret=xmlSchemaValidateDoc(ins.first->second.get(), doc.get()); // validate using compiled/cached schema
  if(ret!=0) throw runtime_error("Failed to validate XML file, see above messages.");
}

void embed(TiXmlElement *&e, const bfs::path &nslocation, map<string,string> &nsprefix, vector<bfs::path> &dependencies, OctEval &octEval) {
  try {
    if(e->ValueStr()==MBXMLUTILSPVNS"embed") {
      NewParamLevel newParamLevel(octEval);
      // check if only href OR child element (This is not checked by the schema)
      TiXmlElement *l=0, *dummy;
      for(dummy=e->FirstChildElement(); dummy!=0; l=dummy, dummy=dummy->NextSiblingElement());
      if((e->Attribute("href") && l && l->ValueStr()!=MBXMLUTILSPVNS"localParameter") ||
         (e->Attribute("href")==0 && (l==0 || l->ValueStr()==MBXMLUTILSPVNS"localParameter")))
        throw TiXmlException("Only the href attribute OR a child element (expect pv:localParameter) is allowed in embed!", e);
      // check if attribute count AND counterName or none of both
      if((e->Attribute("count")==0 && e->Attribute("counterName")!=0) ||
         (e->Attribute("count")!=0 && e->Attribute("counterName")==0))
        throw TiXmlException("Only both, the count and counterName attribute must be given or none of both!", e);
  
      // get file name if href attribute exist
      bfs::path file;
      if(e->Attribute("href")) {
        file=fixPath(TiXml_GetElementWithXmlBase(e,0)->Attribute("xml:base"), e->Attribute("href"));
        dependencies.push_back(file);
      }
  
      // evaluate count using parameters
      long count=1;
      if(e->Attribute("count"))
        count=OctEval::cast<long>(octEval.eval(e, "count"));
  
      // couter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(e->Attribute("counterName"))
        counterName=e->Attribute("counterName");
  
      boost::shared_ptr<TiXmlDocument> enewdoc;
      boost::shared_ptr<TiXmlElement> enewGuard;
      TiXmlElement *enew;
      // validate/load if file is given
      if(!file.empty()) {
        validate(nslocation, file);
        cout<<"Read "<<file<<endl;
        enewdoc.reset(new TiXmlDocument);
        enewdoc->LoadFile(file.generic_string().c_str()); TiXml_PostLoadFile(enewdoc.get());
        enew=enewdoc->FirstChildElement();
        map<string,string> dummy;
        incorporateNamespace(enew, nsprefix, dummy, &dependencies);
      }
      else { // or take the child element (as a clone, because the embed element is deleted)
        if(e->FirstChildElement()->ValueStr()==MBXMLUTILSPVNS"localParameter")
          enewGuard.reset((TiXmlElement*)e->FirstChildElement()->NextSiblingElement()->Clone());
        else
          enewGuard.reset((TiXmlElement*)e->FirstChildElement()->Clone());
        enew=enewGuard.get();
        enew->SetAttribute("xml:base", TiXml_GetElementWithXmlBase(e,0)->Attribute("xml:base")); // add a xml:base attribute
      }
  
      // include a processing instruction with the line number of the original element
      TiXmlUnknown embedLine;
      embedLine.SetValue("?OriginalElementLineNr "+TiXml_itoa(e->Row())+"?");
      enew->InsertBeforeChild(enew->FirstChild(), embedLine);
  
      // generate local paramter for embed
      if(e->FirstChildElement() && e->FirstChildElement()->ValueStr()==MBXMLUTILSPVNS"localParameter") {
        // check if only href OR p:parameter child element (This is not checked by the schema)
        if((e->FirstChildElement()->Attribute("href") && e->FirstChildElement()->FirstChildElement()) ||
           (!e->FirstChildElement()->Attribute("href") && !e->FirstChildElement()->FirstChildElement()))
          throw TiXmlException("Only the href attribute OR the child element p:parameter) is allowed.", e);
        cout<<"Generate local octave parameter string for "<<(file.empty()?"<inline element>":file)<<endl;
        if(e->FirstChildElement()->FirstChildElement()) // inline parameter
          octEval.addParamSet(e->FirstChildElement()->FirstChildElement());
        else { // parameter from href attribute
          bfs::path paramFile=fixPath(TiXml_GetElementWithXmlBase(e,0)->Attribute("xml:base"), e->FirstChildElement()->Attribute("href"));
          // add local parameter file to dependencies
          dependencies.push_back(paramFile);
          // validate local parameter file
          validate(SCHEMADIR/"http___openmbv_berlios_de_MBXMLUtils"/"parameter.xsd", paramFile);
          // read local parameter file
          cout<<"Read local parameter file "<<paramFile<<endl;
          boost::shared_ptr<TiXmlDocument> localparamxmldoc(new TiXmlDocument);
          localparamxmldoc->LoadFile(paramFile.c_str()); TiXml_PostLoadFile(localparamxmldoc.get());
          TiXmlElement *localparamxmlroot=localparamxmldoc->FirstChildElement();
          map<string,string> dummy,dummy2;
          incorporateNamespace(localparamxmlroot,dummy,dummy2,&dependencies);
          // generate local parameters
          octEval.addParamSet(localparamxmlroot);
        }
      }
  
      // delete embed element and insert count time the new element
      for(long i=1; i<=count; i++) {
        octEval.addParam(counterName, i);

        // embed only if 'onlyif' attribute is true
        bool onlyif=true;
        if(e->Attribute("onlyif"))
          onlyif=(OctEval::cast<long>(octEval.eval(e, "onlyif"))==1);
        if(onlyif) {
          cout<<"Embed "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
          if(i==1)
            e=(TiXmlElement*)(e->Parent()->ReplaceChild(e, *enew));
          else
            e=(TiXmlElement*)(e->Parent()->InsertAfterChild(e, *enew));
  
          // include a processing instruction with the count number
          TiXmlUnknown countNr;
          countNr.SetValue("?EmbedCountNr "+TiXml_itoa(i)+"?");
          e->InsertAfterChild(e->FirstChild(), countNr);
      
          // apply embed to new element
          embed(e, nslocation, nsprefix, dependencies, octEval);
        }
        else
          cout<<"Skip embeding "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      return;
    }
    else if(e->ValueStr()==MBXMLUTILSCASADINS"SXFunction")
      return; // skip processing of SXFunction elements
    else {
      octave_value value=octEval.eval(e);
      if(!value.is_empty()) {
        if(e->FirstChildElement())
          e->RemoveChild(e->FirstChildElement());
        else if(e->FirstChildText())
          e->RemoveChild(e->FirstChildText());
        auto_ptr<TiXmlNode> node;
        if(OctEval::getType(value)==OctEval::SXFunctionType)
          node=OctEval::cast<auto_ptr<TiXmlElement> >(value);
        else
          node.reset(new TiXmlText(OctEval::cast<string>(value)));
        e->LinkEndChild(node.release());
      }

      // THIS IS A WORKAROUND! Actually not all 'name' and 'ref*' attributes should be converted but only the
      // XML attributes of a type devived from pv:fullOctaveString and pv:partialOctaveString. But for that a
      // schema aware processor is needed!
      for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next())
        if(a->Name()==string("name") || string(a->Name()).substr(0,3)=="ref") {
          octave_value value=octEval.eval(e, a->Name(), false);
          string s=OctEval::cast<string>(value);
          if(OctEval::getType(value)==OctEval::StringType)
            s=s.substr(1, s.length()-2);
          a->SetValue(s);
        }
    }
  
    // walk tree
    TiXmlElement *c=e->FirstChildElement();
    while(c) {
      embed(c, nslocation, nsprefix, dependencies, octEval);
      if(c==NULL) break;
      c=c->NextSiblingElement();
    }
  }
  catch(const TiXmlException &ex) {
    throw;
  }
  catch(const exception &ex) {
    throw TiXmlException(ex.what(), e);
  }
}

int main(int argc, char *argv[]) {
  try {
    // convert argv to list
    list<string> arg;
    for(int i=1; i<argc; i++)
      arg.push_back(argv[i]);

    // help message
    if(arg.size()<3) {
      cout<<"Usage:"<<endl
          <<"mbxmlutilspp [--dependencies <dep-file-name>] [--mpath <dir> [--mpath <dir> ]]"<<endl
          <<"              <param-file> [dir/]<main-file> <namespace-location-of-main-file>"<<endl
          <<"             [<param-file> [dir/]<main-file> <namespace-location-of-main-file>]"<<endl
          <<"             ..."<<endl
          <<""<<endl
          <<"  --dependencies    Write a newline separated list of dependent files including"<<endl
          <<"                    <param-file> and <main-file> to <dep-file-name>"<<endl
          <<"  --mpath           Add <dir> to the octave search path for m-files"<<endl
          <<""<<endl
          <<"  The output file is named '.pp.<main-file>'."<<endl
          <<"  Use 'none' if not <param-file> is avaliabel."<<endl
          <<""<<endl
          <<"Copyright (C) 2009 Markus Friedrich <friedrich.at.gc@googlemail.com>"<<endl
          <<"This is free software; see the source for copying conditions. There is NO"<<endl
          <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
          <<""<<endl
          <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl;
      return 0;
    }

    // a global oct evaluator just to prevent multiple init/deinit calles
    OctEval globalOctEval;

    // check for environment variables (none default installation)
    SCHEMADIR=bfs::path(getInstallPath())/"share"/"mbxmlutils"/"schema";

    // preserve whitespace and newline in TiXmlText nodes
    TiXmlBase::SetCondenseWhiteSpace(false);

    list<string>::iterator i, i2;

    // dependency file
    vector<bfs::path> dependencies;
    bfs::path depFileName;
    if((i=std::find(arg.begin(), arg.end(), "--dependencies"))!=arg.end()) {
      i2=i; i2++;
      depFileName=(*i2);
      arg.erase(i); arg.erase(i2);
    }

    // mpath
    do {
      if((i=std::find(arg.begin(), arg.end(), "--mpath"))!=arg.end()) {
        i2=i; i2++;
        // the search path is global: use absolute path
        bfs::path absmpath=bfs::absolute(*i2);
        // add to octave search path
        OctEval::addPath(absmpath);
        // add m-files in mpath dir to dependencies
        addFilesInDir(dependencies, absmpath, ".m");
        arg.erase(i); arg.erase(i2);
      }
    }
    while(i!=arg.end());

    // loop over all files
    while(arg.size()>0) {
      // initialize the parameter stack (clear ALL caches)
      OctEval octEval(&dependencies);

      bfs::path paramxml(*arg.begin()); arg.erase(arg.begin());
      bfs::path mainxml(*arg.begin()); arg.erase(arg.begin());
      bfs::path nslocation(*arg.begin()); arg.erase(arg.begin());

      // validate parameter file
      if(paramxml!="none")
        validate(SCHEMADIR/"http___openmbv_berlios_de_MBXMLUtils"/"parameter.xsd", paramxml);

      // read parameter file
      boost::shared_ptr<TiXmlDocument> paramxmldoc;
      if(paramxml!="none") {
        cout<<"Read "<<paramxml<<endl;
        paramxmldoc.reset(new TiXmlDocument);
        paramxmldoc->LoadFile(paramxml.generic_string()); TiXml_PostLoadFile(paramxmldoc.get());
        map<string,string> dummy,dummy2;
        dependencies.push_back(paramxml);
        incorporateNamespace(paramxmldoc->FirstChildElement(),dummy,dummy2,&dependencies);
      }

      // generate octave parameter string
      if(paramxmldoc.get()) {
        cout<<"Generate octave parameter set from "<<paramxml<<endl;
        octEval.addParamSet(paramxmldoc->FirstChildElement());
      }

      // validate main file
      validate(nslocation, mainxml);

      // read main file
      cout<<"Read "<<mainxml<<endl;
      boost::shared_ptr<TiXmlDocument> mainxmldoc(new TiXmlDocument);
      mainxmldoc->LoadFile(mainxml.generic_string()); TiXml_PostLoadFile(mainxmldoc.get());
      map<string,string> nsprefix, dummy;
      dependencies.push_back(mainxml);
      incorporateNamespace(mainxmldoc->FirstChildElement(),nsprefix,dummy,&dependencies);

      // embed/validate/toOctave/unit/eval files
      TiXmlElement *mainxmlele=mainxmldoc->FirstChildElement();
      embed(mainxmlele, nslocation,nsprefix,dependencies, octEval);

      // save result file
      bfs::path mainxmlpp=".pp."+mainxml.filename().generic_string();
      cout<<"Save preprocessed file "<<mainxml<<" as "<<mainxmlpp<<endl;
      TiXml_addLineNrAsProcessingInstruction(mainxmldoc->FirstChildElement());
      unIncorporateNamespace(mainxmldoc->FirstChildElement(), nsprefix);
      mainxmldoc->SaveFile(mainxmlpp.generic_string());

      // validate preprocessed file
      validate(nslocation, mainxmlpp);
    }

    // output dependencies?
    if(!depFileName.empty()) {
      ofstream dependenciesFile(depFileName.generic_string().c_str());
      for(vector<bfs::path>::iterator it=dependencies.begin(); it!=dependencies.end(); it++)
        dependenciesFile<<it->generic_string()<<endl;
    }
  }
  catch(const exception &ex) {
    cerr<<ex.what()<<endl;
    return 1;
  }
  catch(...) {
    cerr<<"Unknown exception."<<endl;
    return 1;
  }
  return 0;
}
