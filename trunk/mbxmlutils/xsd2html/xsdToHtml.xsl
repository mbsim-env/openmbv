<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <xsl:param name="PROJECT"/>
  <xsl:param name="PHYSICALVARIABLEHTMLDOC"/>



  <!-- output method -->
  <xsl:output method="html"
    doctype-public="-//W3C//DTD XHTML 1.0 Strict//EN"
    doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <xsl:template match="/">
    <!-- html header -->
    <html xml:lang="en" lang="en"><head><title><xsl:value-of select="$PROJECT"/> - XML Documentation</title></head><body>
    <h1><xsl:value-of select="$PROJECT"/> - XML Documentation</h1>
    <p>This is the Documentation of the XML representation for <xsl:value-of select="$PROJECT"/>.</p>
    <h2>Contents</h2>
    <ul>
      <li><a name="content-nomenclature" href="#nomenclature">Nomenclature</a></li>
      <li>Elements
        <ul>
          <xsl:for-each select="/xs:schema/xs:element">
            <xsl:sort select="@name"/>
            <li>
              <a style="font-family:monospace;font-weight:bold"><xsl:attribute name="name">content-<xsl:value-of select="@name"/></xsl:attribute>
                <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>&lt;<xsl:value-of select="@name"/>&gt;</a>
            </li>
          </xsl:for-each>
        </ul>
      </li>
    </ul>
    <h2><a name="nomenclature" href="#content-nomenclature">Nomenclature:</a></h2>
    <h3>A element:</h3>
    <p><span style="font-family:monospace;font-weight:bold">&lt;ElementName&gt;</span> [0-2] (Type: <span style="font-family:monospace">elementType</span>)</p>
    <p style="padding-left:3ex;margin:0;margin-bottom:1ex">
      Documentation of the element.
    </p>
    <p>The upper nomenclature defines a XML element named <span style="font-family:monospace">ElementName</span> with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type <span style="font-family:monospace">elementType</span>.<br/>
    A occurance of [optional] means [0-1].</p>
    <h3>A choice of element:</h3>
    <ul style="list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex">
      <li style="color:blue">[1-2]</li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetA&gt;</span></li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.<br/>
    A occurance of [optional] means [0-1].</p>
    <h3>A seqence of elements:</h3>
    <ul style="list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex">
      <li style="color:red">[0-3]</li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetA&gt;</span></li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.<br/>
    A occurance of [optional] means [0-1].</p>
    <h3>Nested sequences/choices:</h3>
    <ul style="list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex">
      <li style="color:red">[1-2]</li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetA&gt;</span></li>
      <li>
        <ul style="list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex">
          <li style="color:blue">[0-3]</li>
          <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetC&gt;</span></li>
          <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetD&gt;</span></li>
        </ul>
      </li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>Sequences and choices can be nested like above.</p>
    <h3>Child Elements:</h3>
    <ul style="list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex">
      <li style="color:red">[1-2]</li>
      <li><span style="font-family:monospace;font-weight:bold">&lt;ParantElemenet&gt;</span>
        <ul style="list-style-type:none;padding-left:4ex">
          <li>
            <ul style="list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex">
              <li style="color:blue">[0-3]</li>
              <li><span style="font-family:monospace;font-weight:bold">&lt;ChildElemenetA&gt;</span></li>
              <li><span style="font-family:monospace;font-weight:bold">&lt;ChildElemenetB&gt;</span></li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
    <p>A indent indicates child elements for a given element.</p>

    <h2>Elements</h2>
    <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:element">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
    </body></html>
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    <h3 style="font-family:monospace;font-weight:bold">
      <a>
        <xsl:attribute name="name">
          <xsl:value-of select="@name"/>
        </xsl:attribute>
        <xsl:attribute name="href">#content-<xsl:value-of select="@name"/></xsl:attribute>
        &lt;<xsl:value-of select="@name"/>&gt;
      </a>
    </h3>
    <!-- abstract -->
    <xsl:if test="@abstract='true'">
      <p>This element ist abstract.</p>
    </xsl:if>
    <!-- inherits -->
    <xsl:if test="@substitutionGroup">
      <p>
        Inherits:
        <a style="font-family:monospace;font-weight:bold">
          <xsl:attribute name="href">#<xsl:value-of select="@substitutionGroup"/></xsl:attribute>
          &lt;<xsl:value-of select="@substitutionGroup"/>&gt;</a>
      </p>
    </xsl:if>
    <!-- inherited by -->
    <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])>0">
      <p>
        Inherited by:
        <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
          <xsl:sort select="@name"/>
          <a style="font-family:monospace;font-weight:bold">
            <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>
            &lt;<xsl:value-of select="@name"/>&gt;</a><xsl:if test="position()!=last()">, </xsl:if>
        </xsl:for-each>
      </p>
    </xsl:if>
    <!-- used in -->
    <p>Can be used in:
      <xsl:apply-templates mode="USEDIN2" select="."/>
    </p>
    <!-- class documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    <!-- XXXXXXXXXXXXXXXX -->
    <!-- attributes -->
    <!--<xsl:apply-templates select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute"/>-->
    <!-- XXXXXXXXXXXXXXXX -->
    <!-- child elements -->
    <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:choice|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
      <p>Child Elements:</p>
      <!-- child elements for not base class -->
      <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
      <!-- child elements for base class -->
      <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
        <ul style="list-style-type:none;padding:0">
          <xsl:apply-templates mode="SIMPLECONTENT" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
            <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
          </xsl:apply-templates>
        </ul>
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <!-- XXXXXXXXXXXXXXXX -->
  <!-- attributes -->
<!--  <xsl:template match="/xs:schema/xs:complexType/xs:attribute">
    <p>
      Attribute: <span style="font-family:monospace;font-weight:bold"><xsl:value-of select="@name"/></span>
      <xsl:if test="@use='required'">
        <span style="color:green">Required</span><xsl:text> </xsl:text>
      </xsl:if>
      (Type: <a style="font-family:monospace">
        <xsl:attribute name="href">
          #<xsl:value-of select="@type"/>
        </xsl:attribute>
        <xsl:value-of select="@type"/>
      </a>)
    </p>
  </xsl:template>-->
  <!-- XXXXXXXXXXXXXXXX -->

  <!-- used in -->
  <xsl:template mode="USEDIN2" match="/xs:schema/xs:element">
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
    <a style="font-family:monospace;font-weight:bold">
      <xsl:attribute name="href">#<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/></xsl:attribute>
      &lt;<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>&gt;</a>,
  </xsl:template>

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <xsl:if test="xs:sequence|xs:choice">
      <ul style="list-style-type:none;padding:0">
        <!-- elements from base class -->
        <li>
          All Elements from 
          <a style="font-family:monospace;font-weight:bold">
            <xsl:attribute name="href">#<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/></xsl:attribute>
            &lt;<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>&gt;</a>
        </li>
        <!-- elements from this class -->
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </xsl:if>
  </xsl:template>



  <!-- child elements -->
  <xsl:template mode="SIMPLECONTENT" match="xs:complexType">
    <ul style="list-style-type:none;padding-left:4ex">
      <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice"/>
    </ul>
  </xsl:template>

  <!-- occurance -->
  <xsl:template mode="OCCURANCE" match="xs:sequence|xs:choice|xs:element">
    <xsl:param name="ELEMENTNAME"/>
    <xsl:param name="COLOR"/>
    <xsl:if test="@minOccurs|@maxOccurs">
      <xsl:element name="{$ELEMENTNAME}">
        <xsl:attribute name="style">
          color:<xsl:value-of select="$COLOR"/>
        </xsl:attribute>
        <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
          [optional]
        </xsl:if>
        <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
          [<xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>]
        </xsl:if>
      </xsl:element>
    </xsl:if>
  </xsl:template>

  <!-- sequence -->
  <xsl:template mode="SIMPLECONTENT" match="xs:sequence">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul style="list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLOR" select="'red'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- choice -->
  <xsl:template mode="SIMPLECONTENT" match="xs:choice">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul style="list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLOR" select="'blue'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
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
        <span style="font-family:monospace;font-weight:bold">&lt;<xsl:value-of select="@name"/>&gt;</span>
      </xsl:if>
      <!-- name by ref -->
      <xsl:if test="@ref">
        <a style="font-family:monospace;font-weight:bold">
          <xsl:attribute name="href">#<xsl:value-of select="@ref"/></xsl:attribute>
          &lt;<xsl:value-of select="@ref"/>&gt;</a>
      </xsl:if><xsl:text> </xsl:text>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select=".">
        <xsl:with-param name="ELEMENTNAME" select="'span'"/>
      </xsl:apply-templates>
      <!-- type -->
      <xsl:if test="@type">
        <!-- type {http://openmbv.berlios.de/MBXMLUTILS/physicalvariable}* -->
        <xsl:if test="substring(@type,1,3)='pv:'">
          (Type: <a style="font-family:monospace">
             <!-- set href to $PHYSICALVARIABLEHTMLDOC#[scalartype|vectortype|matrixtype] -->
            <xsl:attribute name="href"><xsl:value-of select="$PHYSICALVARIABLEHTMLDOC"/>#<xsl:if test="substring(@type,string-length(@type)-5,6)='Scalar'">scalartype</xsl:if>
              <xsl:if test="substring(@type,string-length(@type)-5,6)='Vector'">vectortype</xsl:if>
              <xsl:if test="substring(@type,string-length(@type)-5,6)='Matrix'">matrixtype</xsl:if>
            </xsl:attribute>
            <xsl:value-of select="@type"/>
          </a>)
        </xsl:if>
        <!-- type not {http://openmbv.berlios.de/MBXMLUtils/physicalvariable}* -->
        <xsl:if test="substring(@type,1,3)!='pv:'">
          (Type: <span style="font-family:monospace">
            <xsl:value-of select="@type"/>
          </span>)
        </xsl:if>
      </xsl:if>
      <!-- documentation -->
      <xsl:apply-templates mode="ELEMENTANNOTATION" select="xs:annotation/xs:documentation"/>
      <!-- children -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:complexType"/>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <p>
      <xsl:if test="@source='doxygen'">
        <xsl:attribute name="style">
          color:gray
        </xsl:attribute>
        Doxygen:
      </xsl:if>
      <xsl:value-of select="."/>
    </p>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    <p style="padding-left:3ex;margin:0;margin-bottom:1ex">
      <xsl:if test="@source='doxygen'">
        <xsl:attribute name="style">
          padding-left:3ex;margin:0;margin-bottom:1ex;color:gray
        </xsl:attribute>
        Doxygen:
      </xsl:if>
      <xsl:value-of select="."/>
    </p>
  </xsl:template>

</xsl:stylesheet>
