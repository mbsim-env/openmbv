<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  version="1.0">

  <xsl:param name="PROJECT"/>
  <xsl:param name="DATETIME"/>
  <xsl:param name="MBXMLUTILSVERSION"/>



  <!-- output method -->
  <xsl:output method="html" encoding="UTF-8"/>

  <!-- no default text -->
  <xsl:template match="text()"/>


  <!-- all nodes of all imported schemas and myself -->
  <xsl:param name="ALLNODES" select="document(/xs:schema/xs:import/@schemaLocation)|/"/>



  <xsl:template match="/">
    <!-- html header -->
    <xsl:text disable-output-escaping='yes'>&lt;!DOCTYPE html>
</xsl:text>
    <html lang="en">
    <head>
      <title><xsl:value-of select="$PROJECT"/> - XML Documentation</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css"/>
      <link rel="shortcut icon" href="data:image/x-icon;," type="image/x-icon"/>
      <!-- Note: all defined class names and function names here start with _ to differentiate them from bootstrap ones -->
      <style type="text/css">
        table._figure { }
        *._type { font-family:monospace; }
        *._element { font-family:monospace; font-weight:bold; }
        *._linkpointer { cursor:pointer; }
        *._displaynone { display:none; }
        *._hr { margin-top: 40px; margin-bottom: 40px; }
        *._attribute { font-family:monospace; font-weight:bold; margin-left:2ex; }
        *._attributeNoMargin { font-family:monospace; font-weight:bold; }
        img._eqn { display:block; margin-bottom:1ex; margin-top:1ex; }

        ul._elementsofclass { list-style-type:none; padding:0; }
        ul._elementchild { list-style-type:none; padding-left:4ex; }

        ul._elementchoice { list-style-type:none; border-left-style:solid; border-left-color:#5e58db;
                            padding:0.1ex; margin-top:0.25ex; margin-bottom:0.25ex; }
        ul._elementsequence { list-style-type:none; border-left-style:solid; border-left-color:#db5858;
                              padding:0.1ex; margin-top:0.25ex; margin-bottom:0.25ex; }

        *._badgechoice { background-color:#5e58db; }
        *._badgesequence { background-color:#db5858; }

        *._elementdocuall { padding-left:3ex; margin:0; margin-bottom:1ex; }
        *._elementdocu { }
        *._classdocu { }

        ul._content { padding-left:3ex; list-style-type:none; }
        caption._caption { caption-side:bottom; }
      </style>
      <script type="text/javascript" src="https://cdn.datatables.net/s/bs-3.3.5/jq-2.1.4,dt-1.10.10/datatables.min.js"> </script>
      <script type="text/javascript">
        <![CDATA[
        $(document).ready(function() {
          $("._expandcollapsecontent").click(function() {
            $($(this).parent().children("ul")[0]).toggleClass("_displaynone");
            $(this).toggleClass("glyphicon-expand");
            $(this).toggleClass("glyphicon-collapse-up");
          });
          $("._expandcollapsedoxygen").click(function() {
            $(this).next().toggleClass("_displaynone");
          });
        });
        ]]>
      </script>
    </head>
    <body style="margin:0.5em">
    <div class="page-header">
      <h1><xsl:value-of select="$PROJECT"/> - XML Documentation</h1>
      <p>XML-Namespace: <span class="label label-warning"><xsl:value-of select="/xs:schema/@targetNamespace"/></span></p>
    </div>
    <div class="h2">Contents</div>
    <ul class="_content">
      <li>1 <a id="introduction-content" href="#introduction">Introduction</a></li>
      <li>2 <a id="nomenclature-content" href="#nomenclature">Nomenclature</a>
        <ul class="_content">
          <li>2.1 <a id="legend-content" href="#legend">Legend</a></li>
          <li>2.2 <a id="aelement-content" href="#aelement">An element</a></li>
          <li>2.3 <a id="achoice-content" href="#achoice">A choice of element</a></li>
          <li>2.4 <a id="asequence-content" href="#asequence">A sequence of elements</a></li>
          <li>2.5 <a id="nested-content" href="#nested">Nested sequences/choices</a></li>
          <li>2.6 <a id="childelements-content" href="#childelements">Child Elements</a></li>
        </ul>
      </li>
      <li>3 <a id="elements-content" href="#elements">Elements</a>
        <ul class="_content">
          <xsl:for-each select="/xs:schema/xs:element/@substitutionGroup[not(.=/xs:schema/xs:element/@name) and not(.=preceding::*/@substitutionGroup)]">
            <xsl:sort select="."/>
            <li><a class="_expandcollapsecontent glyphicon glyphicon-collapse-up _linkpointer"></a><a class="_element">
              <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="."/></xsl:attribute>
              &lt;<xsl:value-of select="."/>&gt;</a>
              <ul class="_content">
                <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[@substitutionGroup=current()]">
                  <xsl:with-param name="LEVEL" select="0"/>
                  <xsl:with-param name="LEVELNR" select="'3'"/>
                  <xsl:sort select="@name"/>
                </xsl:apply-templates>
              </ul>
            </li>
          </xsl:for-each>
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[not(@substitutionGroup)]">
            <xsl:with-param name="LEVEL" select="0"/>
            <xsl:with-param name="LEVELNR" select="'3'"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ul>
      </li>
      <li>4 <a id="simpletypes-content" href="#simpletypes">Simple Types</a>
        <xsl:if test="/xs:schema/xs:simpleType">
          <ul class="_content">
            <xsl:for-each select="/xs:schema/xs:simpleType">
              <xsl:sort select="@name"/>
              <li><a class="glyphicon glyphicon-unchecked _linkpointer"/><a class="label label-info _type" id="{@name}-content" href="#{@name}"><xsl:value-of select="@name"/></a></li>
            </xsl:for-each>
          </ul>
        </xsl:if>
      </li>
    </ul>
    <hr class="_hr"/>
    <h1>1 <a id="introduction" href="#introduction-content">Introduction</a></h1>
    <xsl:apply-templates mode="CLASSANNOTATION" select="/xs:schema/xs:annotation/xs:documentation"/>
    <h1>2 <a id="nomenclature" href="#nomenclature-content">Nomenclature</a></h1>
    <h2>2.1 <a id="legend" href="#legend-content">Legend</a></h2>
    <table class="table table-condensed">
      <thead>
        <tr><th>Icon</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><span class="_element">&lt;element&gt;</span></td><td>A XML element of name 'element'</td></tr>
        <tr><td><span class="_attributeNoMargin">attrName</span></td><td>A XML attribute of name 'attrName'</td></tr>
        <tr><td><span class="label label-warning">namespace</span></td><td>A XML namespace of name 'namespace'</td></tr>
        <tr><td><span class="label label-info">type</span></td><td>A XML element or attribute type of name 'type'</td></tr>
        <tr><td><span class="badge progress-bar-success">required</span></td><td>A required XML attribute</td></tr>
        <tr><td><span class="badge">0-2</span></td><td>A occurance of XML elements or attributes</td></tr>
      </tbody>
    </table>
    <h2>2.2 <a id="aelement" href="#aelement-content">An element</a></h2>
    <p><span class="_element">&lt;ElementName&gt;</span><xsl:text> </xsl:text><span class="badge">0-2</span><xsl:text> </xsl:text><span class="label label-info _type">elementType</span>
    <br/><span class="_attribute">attrName1</span><xsl:text> </xsl:text><span class="badge progress-bar-success">required</span><xsl:text> </xsl:text><span class="label label-info _type">typeOfTheAttribute</span>
    <br/><span class="_attribute">attrName2</span><xsl:text> </xsl:text><span class="badge">optional</span><xsl:text> </xsl:text><span class="label label-info _type">typeOfTheAttribute</span></p>
    <p class="_elementdocuall">
      Documentation of the element.
    </p>
    <p>The upper nomenclature defines a XML element named <span class="_element">ElementName</span> with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type <span class="label label-info _type">elementType</span>.<br/>
    A occurance of <span class="badge">optional</span> means <span class="badge">0-1</span>.<br/>
    The element has two attributes named <span class="_attributeNoMargin">attrName1</span> and <span class="_attributeNoMargin">attrName2</span> of type <span class="label label-info _type">typeOfTheAttribute</span>. A attribute can be optional or required.</p>
    <h2>2.3 <a id="achoice" href="#achoice-content">A choice of element</a></h2>
    <ul class="_elementchoice">
      <li><span class="badge _badgechoice">1-2</span></li>
      <li><span class="_element">&lt;ElemenetA&gt;</span></li>
      <li><span class="_element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.<br/>
    A occurance of <span class="badge _badgechoice">optional</span> means <span class="badge _badgechoice">0-1</span>.</p>
    <h2>2.4 <a id="asequence" href="#asequence-content">A sequence of elements</a></h2>
    <ul class="_elementsequence">
      <li><span class="badge _badgesequence">0-3</span></li>
      <li><span class="_element">&lt;ElemenetA&gt;</span></li>
      <li><span class="_element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.<br/>
    A occurance of <span class="badge _badgesequence">optional</span> means <span class="badge _badgesequence">0-1</span>.</p>
    <h2>2.5 <a id="nested" href="#nested-content">Nested sequences/choices</a></h2>
    <ul class="_elementsequence">
      <li><span class="badge _badgesequence">1-2</span></li>
      <li><span class="_element">&lt;ElemenetA&gt;</span></li>
      <li>
        <ul class="_elementchoice">
          <li><span class="badge _badgechoice">0-3</span></li>
          <li><span class="_element">&lt;ElemenetC&gt;</span></li>
          <li><span class="_element">&lt;ElemenetD&gt;</span></li>
        </ul>
      </li>
      <li><span class="_element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>Sequences and choices can be nested like above.</p>
    <h2>2.6 <a id="childelements" href="#childelements-content">Child Elements</a></h2>
    <ul class="_elementsequence">
      <li><span class="badge _badgesequence">1-2</span></li>
      <li><span class="_element">&lt;ParantElemenet&gt;</span>
        <ul class="_elementchild">
          <li>
            <ul class="_elementchoice">
              <li><span class="badge _badgechoice">0-3</span></li>
              <li><span class="_element">&lt;ChildElemenetA&gt;</span></li>
              <li><span class="_element">&lt;ChildElemenetB&gt;</span></li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
    <p>A indent indicates child elements for a given element.</p>

    <h1>3 <a id="elements" href="#elements-content">Elements</a></h1>
    <xsl:for-each select="/xs:schema/xs:element/@substitutionGroup[not(.=/xs:schema/xs:element/@name) and not(.=preceding::*/@substitutionGroup)]">
      <xsl:sort select="."/>
      <!-- heading -->
      <!-- use h2 for all section headings independent of the LEVEL -->
      <div class="h2 _element"><a>
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="."/></xsl:attribute>
        &lt;<xsl:value-of select="."/>&gt;</a></div>
      <p>This element is defined by the XML Schema (Project) with the namespace
        <span class="label label-warning"><xsl:value-of select="../namespace::*[name()=substring-before(current(),':')]"/></span>, which is
        included by this XML Schema (Project). See the documentation of the included XML Schema (Project) for this element.</p>
      <hr class="_hr"/>
      <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=current()]">
        <xsl:with-param name="LEVEL" select="1"/>
        <xsl:with-param name="LEVELNR" select="'3'"/>
        <xsl:sort select="@name"/>
      </xsl:apply-templates>
    </xsl:for-each>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[not(@substitutionGroup)]">
      <xsl:with-param name="LEVEL" select="0"/>
      <xsl:with-param name="LEVELNR" select="'3'"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    <h1>4 <a id="simpletypes" href="#simpletypes-content">Simple Types</a></h1>
    <xsl:apply-templates mode="SIMPLETYPE" select="/xs:schema/xs:simpleType">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    <span class="pull-left small"><a href="http://www.mbsim-env.de/mbsim/html/impressum_disclaimer_datenschutz.html#impressum">Impressum</a> /
    <a href="http://www.mbsim-env.de/mbsim/html/impressum_disclaimer_datenschutz.html#disclaimer">Disclaimer</a> /
    <a href="http://www.mbsim-env.de/mbsim/html/impressum_disclaimer_datenschutz.html#datenschutz">Datenschutz</a></span><span class="pull-right small">
    Generated on <xsl:value-of select="$DATETIME"/> for <xsl:value-of select="$PROJECT"/> by MBXMLUtils <a href="http://validator.w3.org/check?uri=referer"><img src="https://www.w3.org/Icons/valid-html401-blue.png" alt="Valid HTML"/></a></span>
    </body></html>
  </xsl:template>

  <!-- generate html link form a attribute -->
  <xsl:template mode="GENLINK" match="@*">
    <xsl:param name="V1" select="../namespace::*[name()=substring-before(current(),':')]"/>
    <xsl:param name="V2" select="translate($V1,'.:/','___')"/>
    <xsl:text>../</xsl:text><xsl:value-of select="$V2"/><xsl:text>/index.html#</xsl:text>
    <xsl:if test="not(contains(.,':'))">
      <xsl:value-of select="."/>
    </xsl:if>
    <xsl:if test="contains(.,':')">
      <xsl:value-of select="substring-after(.,':')"/>
    </xsl:if>
  </xsl:template>

  <!-- generate contents -->
  <xsl:template mode="CONTENT" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="LEVELNR"/>
    <xsl:param name="NAME" select="@name"/>
    <li>
      <a>
        <xsl:if test="/xs:schema/xs:element[@substitutionGroup=$NAME]">
          <xsl:attribute name="class">_expandcollapsecontent glyphicon glyphicon-collapse-up _linkpointer</xsl:attribute>
        </xsl:if>
        <xsl:if test="not(/xs:schema/xs:element[@substitutionGroup=$NAME])">
          <xsl:attribute name="class">glyphicon glyphicon-unchecked _linkpointer</xsl:attribute>
        </xsl:if>
      </a>
      <xsl:if test="$LEVEL &lt; 0"><!-- prevent numbers -->
        <xsl:value-of select="$LEVELNR"/>.<xsl:value-of select="position()"/>
      </xsl:if>
      <xsl:text> </xsl:text>
      <a class="_element" id="{@name}-content" href="#{@name}">&lt;<xsl:value-of select="@name"/>&gt;</a>
      <xsl:if test="/xs:schema/xs:element[@substitutionGroup=$NAME]">
        <ul class="_content">
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
            <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
            <xsl:with-param name="LEVELNR" select="concat($LEVELNR,'.',position())"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ul>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- walk throw all elements -->
  <xsl:template mode="WALKCLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="LEVELNR"/>
    <xsl:param name="NAME" select="@name"/>
    <xsl:apply-templates mode="CLASS" select=".">
      <xsl:with-param name="LEVEL" select="$LEVEL"/>
      <xsl:with-param name="TITLENR" select="concat($LEVELNR,'.',position())"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
      <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
      <xsl:with-param name="LEVELNR" select="concat($LEVELNR,'.',position())"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="TITLENR"/>
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    <!-- use h2 for all section headings independent of the LEVEL -->
    <div class="h2 _element">
      <xsl:if test="$LEVEL &lt; 0"><!-- prevent numbers -->
        <xsl:value-of select="$TITLENR"/>
      </xsl:if>
      <xsl:text> </xsl:text>
      <a id="{@name}" href="#{@name}-content">&lt;<xsl:value-of select="@name"/>&gt;</a>
    </div>
    <!-- properties -->
    <div class="panel panel-success">
      <div class="panel-heading">Object properties</div>
      <table class="table table-condensed">
        <tbody>
          <!-- abstract -->
          <tr><td>Abstract Element:</td><td>
            <xsl:if test="@abstract='true'">true</xsl:if>
            <xsl:if test="@abstract!='true' or not(@abstract)">false</xsl:if>
          </td></tr>
          <!-- inherits -->
          <tr><td>Inherits:</td><td>
            <xsl:if test="@substitutionGroup">
              <a class="_element">
                <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@substitutionGroup"/></xsl:attribute>
                &lt;<xsl:value-of select="@substitutionGroup"/>&gt;</a>
            </xsl:if>
          </td></tr>
          <!-- inherited by -->
          <tr><td>Inherited by:</td><td>
            <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])>0">
              <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
                <xsl:sort select="@name"/>
                <a class="_element" href="#{@name}">&lt;<xsl:value-of select="@name"/>&gt;</a>, 
              </xsl:for-each>
            </xsl:if>
          </td></tr>
          <!-- used in -->
          <!--<tr><td>Can be used in:</td><td><xsl:apply-templates mode="USEDIN2" select="."/></td></tr>-->
          <!-- class attributes -->
          <tr><td>Attributes:</td><td>
          <xsl:apply-templates mode="CLASSATTRIBUTE" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:attribute"/>
          </td></tr>
        </tbody>
      </table>
    </div>
    <!-- class documentation -->
    <div class="panel panel-info">
      <div class="panel-heading">Object documentation</div>
      <div class="panel-body">
        <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
      </div>
    </div>
    <!-- child elements -->
    <div class="panel panel-warning">
      <div class="panel-heading">Child Elements</div>
      <div class="panel-body">
        <!-- child elements for not base class -->
        <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
        <!-- child elements for base class -->
        <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
          <ul class="_elementsofclass">
            <xsl:apply-templates mode="SIMPLECONTENT" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
              <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
            </xsl:apply-templates>
          </ul>
        </xsl:if>
      </div>
    </div>

    <!-- BEGIN show example xml code -->
    <xsl:if test="not(@abstract) or @abstract='false'">

      <div class="btn-group btn-group-sm">
        <button type="button" class="btn btn-primary dropdown-toggle" data-toggle="dropdown">
          Show example <span class="caret"></span>
        </button>
        <pre class="dropdown-menu" role="menu">
          <xsl:apply-templates mode="EXAMPLEELEMENT" select=".">
            <xsl:with-param name="FULLNAME" select="concat('{',namespace::*[name()=substring-before(current()/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))"/>
            <!-- this FULLNAME is equal to select="@name" with full namespace awareness -->
            <xsl:with-param name="INDENT" select="''"/>
            <xsl:with-param name="CURRENTNS" select="''"/>
          </xsl:apply-templates><xsl:text>
</xsl:text>
          <xsl:apply-templates mode="EXAMPLECHILDS" select=".">
            <xsl:with-param name="INDENT" select="'  '"/>
            <xsl:with-param name="CURRENTNS" select="namespace::*[name()=substring-before(current()/@name,':')]"/>
          </xsl:apply-templates>
          <xsl:text>&lt;/</xsl:text><xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/><xsl:text>&gt;
</xsl:text>
        </pre>
      </div>
    </xsl:if>
    <!-- END show example xml code -->
    <hr class="_hr"/>
  </xsl:template>

  <!-- simple type -->
  <xsl:template mode="SIMPLETYPE" match="/xs:schema/xs:simpleType">
    <a class="label label-info _type" id="{@name}" href="#{@name}-content"><xsl:value-of select="@name"/></a>
    <!-- simpleType documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    <hr class="_hr"/>
  </xsl:template>

  <!-- class attributes -->
  <xsl:template mode="CLASSATTRIBUTE" match="/xs:schema/xs:complexType/xs:attribute|/xs:schema/xs:complexType/xs:complexContent/xs:extension/xs:attribute">
    <xsl:if test="@name">
      <span class="_element"><xsl:value-of select="@name"/></span><xsl:text> </xsl:text>
      <xsl:if test="@use='required'">
        <span class="badge progress-bar-success">required</span><xsl:text> </xsl:text>
      </xsl:if>
      <xsl:if test="@use!='required'">
        <span class="badge">optional</span><xsl:text> </xsl:text>
      </xsl:if>
      <a class="label label-info _type">
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@type"/></xsl:attribute>
        <xsl:value-of select="@type"/></a>
      <br/>
    </xsl:if>
    <xsl:if test="@ref">
      <a class="_element">
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@ref"/></xsl:attribute>
        <xsl:value-of select="@ref"/></a><br/>
    </xsl:if>
  </xsl:template>

  <!-- used in -->
  <!--<xsl:template mode="USEDIN2" match="/xs:schema/xs:element">
      <xsl:param name="SUBSTGROUP" select="@substitutionGroup"/>
      <xsl:param name="CLASSNAME" select="@name"/>
      <xsl:apply-templates mode="USEDIN" select="/descendant::xs:element[@ref=$CLASSNAME]"/>
      <xsl:apply-templates mode="USEDIN2" select="/xs:schema/xs:element[@name=$SUBSTGROUP]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:element">
    <xsl:apply-templates mode="USEDIN" select="ancestor::xs:complexType[last()]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:complexType">
    <xsl:param name="CLASSTYPE" select="@name"/>
    <a class="element">
      <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/></xsl:attribute>
      &lt;<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>&gt;</a>,
  </xsl:template>-->

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <ul class="_elementsofclass">
      <!-- elements from base class -->
      <li>
        All Elements from 
        <a class="_element">
          <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/></xsl:attribute>
          &lt;<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>&gt;</a>
      </li>
      <!-- elements from this class -->
      <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    </ul>
  </xsl:template>



  <!-- child elements -->
  <xsl:template mode="SIMPLECONTENT" match="xs:complexType">
    <xsl:if test="xs:sequence|xs:choice">
      <ul class="_elementchild">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice"/>
      </ul>
    </xsl:if>
  </xsl:template>

  <!-- occurance -->
  <xsl:template mode="OCCURANCE" match="xs:sequence|xs:choice">
    <xsl:param name="COLORSTYLE"/>
    <xsl:if test="@minOccurs|@maxOccurs">
      <li><span>
        <xsl:if test="$COLORSTYLE!=''">
          <xsl:attribute name="class">badge <xsl:value-of select="$COLORSTYLE"/></xsl:attribute>
        </xsl:if>
        <xsl:if test="@minOccurs=0 and not(@maxOccurs)">optional</xsl:if>
        <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
          <xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>
        </xsl:if>
      </span></li>
    </xsl:if>
  </xsl:template>
  <xsl:template mode="OCCURANCE" match="xs:element">
    <xsl:if test="@minOccurs|@maxOccurs">
      <span>
        <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
          <span class="badge">optional</span><xsl:text> </xsl:text>
        </xsl:if>
        <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
          <span class="badge"><xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if></span>
        </xsl:if>
      </span>
    </xsl:if>
  </xsl:template>

  <!-- sequence -->
  <xsl:template mode="SIMPLECONTENT" match="xs:sequence">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul class="_elementsequence">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="COLORSTYLE" select="'_badgesequence'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice|xs:any">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- choice -->
  <xsl:template mode="SIMPLECONTENT" match="xs:choice">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul class="_elementchoice">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="COLORSTYLE" select="'_badgechoice'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice|xs:any">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- element -->
  <xsl:template mode="SIMPLECONTENT" match="xs:element">
    <xsl:param name="FUNCTIONNAME" select="@name"/>
    <xsl:param name="CLASSNAME"/>
    <li>
      <!-- name by not(ref) -->
      <xsl:if test="not(@ref)">
        <span class="_element">&lt;<xsl:value-of select="@name"/>&gt;</span>
      </xsl:if>
      <!-- name by ref -->
      <xsl:if test="@ref">
        <a class="_element">
          <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@ref"/></xsl:attribute>
          &lt;<xsl:value-of select="@ref"/>&gt;</a>
      </xsl:if><xsl:text> </xsl:text>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select="."/><xsl:text> </xsl:text>
      <!-- type -->
      <xsl:if test="@type"><a class="label label-info _type">
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@type"/></xsl:attribute>
        <xsl:value-of select="@type"/></a> </xsl:if>
      <!-- element attributes -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="ELEMENTATTRIBUTE" select="xs:complexType/xs:attribute"/>
      </xsl:if>
      <!-- documentation -->
      <div class="_elementdocuall">
        <xsl:apply-templates mode="ELEMENTANNOTATION" select="xs:annotation/xs:documentation"/>
      </div>
      <!-- children -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:complexType"/>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- element attributes -->
  <xsl:template mode="ELEMENTATTRIBUTE" match="xs:attribute">
    <br/>
    <span class="_attribute"><xsl:value-of select="@name"/></span><xsl:text> </xsl:text>
    <xsl:if test="@use='required'">
      <span class="badge progress-bar-success">required</span><xsl:text> </xsl:text>
    </xsl:if>
    <xsl:if test="@use!='required'">
      <span class="badge"> optional</span><xsl:text> </xsl:text>
    </xsl:if>
    <a class="label label-info _type">
      <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@type"/></xsl:attribute>
      <xsl:value-of select="@type"/></a><xsl:text> </xsl:text>
  </xsl:template>

  <!-- any element -->
  <xsl:template mode="SIMPLECONTENT" match="xs:any">
    <xsl:param name="CLASSNAME"/>
    <li>
      <!-- name -->
      <span class="_element">&lt;xs:any&gt;</span>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select="."/><xsl:text> </xsl:text>
      <!-- type -->
      <span class="label label-info _type">xs:any</span><xsl:text> </xsl:text>
      <!-- documentation -->
      <div class="_elementdocuall">
        Any element of the namespace <span class="label label-warning"><xsl:value-of select="@namespace"/></span>
      </div>
    </li>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <!-- add Doxygen documentation dynamically (expand-/colapse-able) if other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])>0">
      <a class="_expandcollapsedoxygen _linkpointer"><i><small>Doxygen <span class="caret"/></small></i></a>
      <div class="_displaynone _classdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
    </xsl:if>
    <!-- add Doxygen documentation staticlly if no other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])=0">
      <div class="_classdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
    </xsl:if>
    <!-- always add other documentation statically -->
    <xsl:if test="@source!='doxygen' or not(@source)">
      <div class="_classdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
    </xsl:if>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    <!-- add Doxygen documentation dynamically (expand-/colapse-able) if other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])>0">
      <a class="_expandcollapsedoxygen _linkpointer"><i><small>Doxygen <span class="caret"/></small></i></a>
      <div class="_displaynone _elementdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
    </xsl:if>
    <!-- add Doxygen documentation staticlly if no other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])=0">
      <div class="_elementdocu">
        <xsl:apply-templates mode="CLONEDOC"/>
      </div>
    </xsl:if>
    <!-- always add other documentation statically -->
    <xsl:if test="@source!='doxygen' or not(@source)">
      <div class="_elementdocu">
        <xsl:apply-templates mode="CLONEDOC"/>
      </div>
    </xsl:if>
  </xsl:template>

  <!-- clone doxygen xml/html part -->
  <xsl:template mode="CLONEDOC" match="*">
    <xsl:copy>
      <xsl:for-each select="@*">
        <xsl:copy/>
      </xsl:for-each>
      <xsl:apply-templates mode="CLONEDOC"/>
    </xsl:copy>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="text()">
    <xsl:copy/>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="object[@class='eqn']">
    <img class="_eqn" src="{concat('_',generate-id(),'.png')}">
      <xsl:attribute name="alt"><xsl:value-of select="."/></xsl:attribute>
    </img>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="object[@class='inlineeqn']">
    <img class="_inlineeqn" src="{concat('_',generate-id(),'.png')}">
      <xsl:attribute name="alt"><xsl:value-of select="."/></xsl:attribute>
    </img>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="object[@class='figure']">
    <table class="_figure">
      <caption class="_caption"><xsl:value-of select="@title"/></caption>
      <tr><td><img src="{@data}.png" alt="{@data}"></img></td></tr>
    </table>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="object[@class='figure_html']">
    <table class="_figure">
      <caption class="_caption"><xsl:value-of select="@title"/></caption>
      <tr><td><img src="{@data}" alt="{@data}"></img></td></tr>
    </table>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="object[@class='figure_latex']">
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="a[@class='link']">
    <a><xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@href"/></xsl:attribute>
    <xsl:value-of select="."/></a>
  </xsl:template>

  <!-- BEGIN show example xml code -->
  <xsl:template mode="EXAMPLEELEMENT" match="/xs:schema/xs:element">
    <xsl:param name="FULLNAME"/>
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:param name="NS_SUBSTITUTIONGROUP" select="namespace::*[name()=substring-before(current()/@substitutionGroup,':')]"/>
    <xsl:param name="NAME_SUBSTITUTIONGROUP" select="translate(substring(@substitutionGroup,string-length(substring-before(@substitutionGroup,':'))+1),':','')"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before(current()/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=$FULLNAME">
      <xsl:value-of select="$INDENT"/>&lt;<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>
    </xsl:if>
    <!-- this apply-templates (includeing the EXAMPLEELEMENT_WITHATTRFQN template) is equal to select=".../[@name=current()/@substitutionGroup]" with namespace aware attributes values -->
    <xsl:apply-templates mode="EXAMPLEELEMENT_WITHATTRFQN" select="$ALLNODES/xs:schema/xs:element">
      <xsl:with-param name="FULLNAME" select="$FULLNAME"/>
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      <xsl:with-param name="FQN" select="concat('{',$NS_SUBSTITUTIONGROUP,'}',$NAME_SUBSTITUTIONGROUP)"/>
      <xsl:with-param name="ATTRNAME" select="'name'"/>
    </xsl:apply-templates>
    <xsl:for-each select="/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:attribute|/xs:schema/xs:complexType[@name=current()/@type]/xs:attribute">
      <xsl:text> </xsl:text><xsl:value-of select="@name"/><xsl:value-of select="@ref"/>="VALUE"<xsl:text></xsl:text>
    </xsl:for-each>
    <xsl:if test="concat('{',namespace::*[name()=substring-before(current()/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=$FULLNAME">
      <xsl:apply-templates mode="XXX" select=".">
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates>&gt;<xsl:text></xsl:text>
    </xsl:if>
  </xsl:template>
  <!-- just required to workaround the issue that many xslt processors (e.g. Xalan) does not provide the correct parent element for
  a namespace node -->
  <xsl:template match="*" mode="EXAMPLEELEMENT_WITHATTRFQN">
    <xsl:param name="FULLNAME"/>
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:param name="FQN"/>
    <xsl:param name="ATTRNAME"/>
    <xsl:param name="ATTR" select="@*[name()=$ATTRNAME]"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before($ATTR,':')],'}',translate(substring($ATTR,string-length(substring-before($ATTR,':'))+1),':',''))=$FQN">
      <xsl:apply-templates select="." mode="EXAMPLEELEMENT">
        <xsl:with-param name="FULLNAME" select="$FULLNAME"/>
        <xsl:with-param name="INDENT" select="$INDENT"/>
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates>
    </xsl:if>
  </xsl:template>
  <xsl:template mode="XXX" match="*">
    <xsl:param name="CURRENTNS"/>
    <xsl:if test="@name">
      <xsl:if test="$CURRENTNS!=namespace::*[name()=substring-before(current()/@name,':')]">
        <xsl:text> xmlns="</xsl:text><xsl:value-of select="namespace::*[name()=substring-before(current()/@name,':')]"/>"<xsl:text></xsl:text>
      </xsl:if>
    </xsl:if>
    <xsl:if test="@ref">
      <xsl:if test="$CURRENTNS!=namespace::*[name()=substring-before(current()/@ref,':')]">
        <xsl:text> xmlns="</xsl:text><xsl:value-of select="namespace::*[name()=substring-before(current()/@ref,':')]"/>"<xsl:text></xsl:text>
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <xsl:template mode="EXAMPLECHILDS" match="/xs:schema/xs:element">
    <xsl:param name="INDENT"/>
    <xsl:param name="NS_SUBSTITUTIONGROUP" select="namespace::*[name()=substring-before(current()/@substitutionGroup,':')]"/>
    <xsl:param name="NAME_SUBSTITUTIONGROUP" select="translate(substring(@substitutionGroup,string-length(substring-before(@substitutionGroup,':'))+1),':','')"/>
    <xsl:param name="CURRENTNS"/>
    <!-- this apply-templates (includeing the EXAMPLECHILDS_WITHATTRFQN template) is equal to select="...[@name=current()/@substitutionGroup]" with namespace aware attributes values -->
    <xsl:apply-templates mode="EXAMPLECHILDS_WITHATTRFQN" select="$ALLNODES/xs:schema/xs:element">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      <xsl:with-param name="FQN" select="concat('{',$NS_SUBSTITUTIONGROUP,'}',$NAME_SUBSTITUTIONGROUP)"/>
      <xsl:with-param name="ATTRNAME" select="'name'"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="EXAMPLELOCAL" select="/xs:schema/xs:complexType[@name=current()/@type]/xs:sequence|/xs:schema/xs:complexType[@name=current()/@type]/xs:choice|/xs:schema/xs:complexType[@name=current()/@type]/xs:element|/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:sequence|/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:choice|/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:element">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
  </xsl:template>
  <!-- just required to workaround the issue that many xslt processors (e.g. Xalan) does not provide the correct parent element for
  a namespace node -->
  <xsl:template match="*" mode="EXAMPLECHILDS_WITHATTRFQN">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:param name="FQN"/>
    <xsl:param name="ATTRNAME"/>
    <xsl:param name="ATTR" select="@*[name()=$ATTRNAME]"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before($ATTR,':')],'}',translate(substring($ATTR,string-length(substring-before($ATTR,':'))+1),':',''))=$FQN">
      <xsl:apply-templates select="." mode="EXAMPLECHILDS">
        <xsl:with-param name="INDENT" select="$INDENT"/>
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates>
    </xsl:if>
  </xsl:template>

  <xsl:template mode="EXAMPLELOCAL" match="xs:sequence">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:apply-templates mode="EXAMPLELOCAL" select="xs:sequence|xs:choice|xs:element">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template mode="EXAMPLELOCAL" match="xs:choice">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:apply-templates mode="EXAMPLELOCAL" select="child::*[position()=1]">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template mode="EXAMPLELOCAL" match="xs:element">
    <xsl:param name="INDENT"/>
    <xsl:param name="NS_REF" select="namespace::*[name()=substring-before(current()/@ref,':')]"/>
    <xsl:param name="NAME_REF" select="translate(substring(@ref,string-length(substring-before(@ref,':'))+1),':','')"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:if test="@ref">
      <!-- this apply-templates (includeing the REFABSTRACT_WITHATTRFQN template) is equal to select=".../[@name=current()/@ref and @abstract='true']" with namespace aware attributes values -->
      <xsl:apply-templates mode="REFABSTRACT_WITHATTRFQN" select="$ALLNODES/xs:schema/xs:element">
        <xsl:with-param name="INDENT" select="$INDENT"/>
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
        <xsl:with-param name="FQN" select="concat('{',$NS_REF,'}',$NAME_REF)"/>
        <xsl:with-param name="ATTRNAME" select="'name'"/>
      </xsl:apply-templates>
      <!-- this apply-templates (includeing the REFNOTABSTRACT_WITHATTRFQN template) is equal to select=".../[@name=current()/@ref and (@abstract='false' or not(@abstract)]" with namespace aware attributes values -->
      <xsl:apply-templates mode="REFNOTABSTRACT_WITHATTRFQN" select="$ALLNODES/xs:schema/xs:element">
        <xsl:with-param name="INDENT" select="$INDENT"/>
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
        <xsl:with-param name="FQN" select="concat('{',$NS_REF,'}',$NAME_REF)"/>
        <xsl:with-param name="ATTRNAME" select="'name'"/>
      </xsl:apply-templates>
    </xsl:if>
    <xsl:if test="@name">
      <xsl:value-of select="$INDENT"/>&lt;<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>
      <xsl:for-each select="xs:complexType/xs:attribute">
        <xsl:text> </xsl:text><xsl:value-of select="@name"/><xsl:value-of select="@ref"/>="VALUE"<xsl:text></xsl:text>
      </xsl:for-each>
      <xsl:if test="@type">
        <xsl:apply-templates mode="XXX" select=".">
          <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
        </xsl:apply-templates>&gt;VALUE&lt;/<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>&gt;<xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/><xsl:text>
</xsl:text>
      </xsl:if>
      <xsl:if test="not(@type)">
        <xsl:if test="xs:complexType/xs:sequence|xs:complexType/xs:choice|xs:complexType/xs:element">
          <xsl:apply-templates mode="XXX" select=".">
            <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
          </xsl:apply-templates>&gt;<xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/><xsl:text>
</xsl:text>
          <xsl:apply-templates mode="EXAMPLELOCAL" select="xs:complexType/xs:sequence|xs:complexType/xs:choice|xs:complexType/xs:element">
            <xsl:with-param name="INDENT" select="concat($INDENT, '  ')"/>
            <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
          </xsl:apply-templates>
          <xsl:value-of select="$INDENT"/>&lt;/<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/><xsl:text>&gt;
</xsl:text>
        </xsl:if>
        <xsl:if test="not(xs:complexType/xs:sequence|xs:complexType/xs:choice|xs:complexType/xs:element)">
          <xsl:apply-templates mode="XXX" select=".">
            <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
          </xsl:apply-templates>/&gt;<xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/>
          <!-- this apply-templates (includeing the EXAMPLEELEMENT_WITHATTRFQN template) is equal to select=".../[@name=current()/@substitutionGroup]" with namespace aware attributes values -->
          <xsl:apply-templates mode="ABSTRACT_WITHATTRFQN" select="$ALLNODES/xs:schema/xs:element">
            <xsl:with-param name="FQN" select="concat('{',$NS_REF,'}',$NAME_REF)"/>
            <xsl:with-param name="ATTRNAME" select="'name'"/>
          </xsl:apply-templates><xsl:text>
</xsl:text>
        </xsl:if>
      </xsl:if>
    </xsl:if>
  </xsl:template>
  <!-- just required to workaround the issue that many xslt processors (e.g. Xalan) does not provide the correct parent element for
  a namespace node -->
  <xsl:template match="*" mode="ABSTRACT_WITHATTRFQN">
    <xsl:param name="FQN"/>
    <xsl:param name="ATTRNAME"/>
    <xsl:param name="ATTR" select="@*[name()=$ATTRNAME]"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before($ATTR,':')],'}',translate(substring($ATTR,string-length(substring-before($ATTR,':'))+1),':',''))=$FQN and @abstract='true'"> &lt;!-- Abstract --&gt;</xsl:if>
  </xsl:template>
  <!-- just required to workaround the issue that many xslt processors (e.g. Xalan) does not provide the correct parent element for
  a namespace node -->
  <xsl:template match="*" mode="REFABSTRACT_WITHATTRFQN">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:param name="FQN"/>
    <xsl:param name="ATTRNAME"/>
    <xsl:param name="ATTR" select="@*[name()=$ATTRNAME]"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before($ATTR,':')],'}',translate(substring($ATTR,string-length(substring-before($ATTR,':'))+1),':',''))=$FQN and @abstract='true'">
      <xsl:value-of select="$INDENT"/>&lt;<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>
      <xsl:apply-templates mode="XXX" select=".">
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates>/&gt; &lt;!-- abstract --&gt;<xsl:text>
</xsl:text>
    </xsl:if>
  </xsl:template>
  <!-- just required to workaround the issue that many xslt processors (e.g. Xalan) does not provide the correct parent element for
  a namespace node -->
  <xsl:template match="*" mode="REFNOTABSTRACT_WITHATTRFQN">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:param name="FQN"/>
    <xsl:param name="ATTRNAME"/>
    <xsl:param name="ATTR" select="@*[name()=$ATTRNAME]"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before($ATTR,':')],'}',translate(substring($ATTR,string-length(substring-before($ATTR,':'))+1),':',''))=$FQN and (@abstract='false' or not(@abstract))">
      <xsl:apply-templates mode="EXAMPLEELEMENT" select=".">
        <xsl:with-param name="FULLNAME" select="concat('{',namespace::*[name()=substring-before(current()/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))"/>
        <!-- this FULLNAME is equal to select="@name" with full namespace awareness -->
        <xsl:with-param name="INDENT" select="$INDENT"/>
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates><xsl:text>
</xsl:text>
      <xsl:apply-templates mode="EXAMPLECHILDS" select=".">
        <xsl:with-param name="INDENT" select="concat($INDENT, '  ')"/>
        <xsl:with-param name="CURRENTNS" select="namespace::*[name()=substring-before(current()/@name,':')]"/>
      </xsl:apply-templates>
      <xsl:value-of select="$INDENT"/>
      <xsl:text>&lt;/</xsl:text>
      <xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/><xsl:text>&gt;
</xsl:text>
    </xsl:if>
  </xsl:template>
  
  <xsl:template mode="EXAMPLEOPTIONAL" match="*">
    <xsl:if test="@minOccurs=0"> &lt;!-- optional --&gt;</xsl:if>
  </xsl:template>
  <!-- END show example xml code -->

</xsl:stylesheet>
