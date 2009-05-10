<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  version="2.0">

  <!-- adds the doxygen class and class member documentation to a xsd-file -->

  <xsl:param name="DOXYGENDOC"/>
  <xsl:param name="DOXYGENCLASSPREFIX"/>
  <xsl:param name="DOXYGENFUNCTIONPREFIX1"/>
  <xsl:param name="DOXYGENFUNCTIONPREFIX2"/>



  <xsl:output method="xml" version="1.0" indent="yes"/>

  <!-- clone xs:schema element; called without param -->
  <xsl:template match="/xs:schema">
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <xsl:apply-templates/>
    </xsl:copy>
  </xsl:template>

  <!-- clone elements; called with param -->
  <xsl:template match="/|*">
    <xsl:param name="CLASSNAME"/>
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <xsl:apply-templates>
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

  <!-- clone attributes -->
  <xsl:template match="@*">
    <xsl:copy/>
  </xsl:template>

  <!-- clone processing-instruction and comment -->
  <xsl:template match="processing-instruction()|comment()">
    <xsl:copy/>
  </xsl:template>





  <!-- overwrite template for classes -->
  <xsl:template match="/xs:schema/xs:element">
    <xsl:param name="CLASSNAME" select="@name"/>
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <!-- create xs:annotation for doxygen if not existing -->
      <xsl:if test="not(xs:annotation)">
        <xs:annotation>
          <xsl:apply-templates select="document($DOXYGENDOC)/doxygen/compounddef/compoundname[.=concat($DOXYGENCLASSPREFIX,$CLASSNAME)]"/>
        </xs:annotation>
      </xsl:if>
      <xsl:apply-templates>
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

  <!-- append doxygen to existing xs:annotation -->
  <xsl:template match="/xs:schema/xs:element/xs:annotation" priority="2">
    <xsl:param name="CLASSNAME"/>
    <xsl:copy>
      <xsl:apply-templates/>
      <xsl:apply-templates select="document($DOXYGENDOC)/doxygen/compounddef/compoundname[.=concat($DOXYGENCLASSPREFIX,$CLASSNAME)]"/>
    </xsl:copy>
  </xsl:template>

  <!-- add doxygen class documentation -->
  <xsl:template match="/doxygen/compounddef/compoundname">
    <xsl:if test="string-length(normalize-space(../briefdescription))!=0 or string-length(normalize-space(../detaileddescription))!=0">
      <xs:documentation source="doxygen">
        <xsl:if test="string-length(normalize-space(../briefdescription))!=0">
          <xsl:apply-templates select="../briefdescription"/>
        </xsl:if>
        <xsl:if test="string-length(normalize-space(../detaileddescription))!=0">
          <xsl:apply-templates select="../detaileddescription"/>
        </xsl:if>
      </xs:documentation>
    </xsl:if>
  </xsl:template>



  <!-- overwrite template for class mebers -->
  <xsl:template match="/xs:schema/xs:complexType">
    <xsl:param name="CLASSTYPE" select="@name"/>
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
<xsl:message>A<xsl:value-of select="$CLASSTYPE"/>A</xsl:message>
<xsl:message>X<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>X</xsl:message>
      <xsl:apply-templates>
        <xsl:with-param name="CLASSNAME" select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="xs:element">
    <xsl:param name="FUNCTIONNAME" select="concat(upper-case(substring(@name,1,1)),substring(@name,2))"/><!-- set first character upper case -->
    <xsl:param name="CLASSNAME"/>
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <!-- create xs:annotation for doxygen if not existing -->
      <xsl:if test="not(xs:annotation)">
        <xs:annotation>
          <xsl:apply-templates select="document($DOXYGENDOC)/doxygen/compounddef/compoundname[.=concat($DOXYGENCLASSPREFIX,$CLASSNAME)]/../sectiondef/memberdef/name[.=concat($DOXYGENFUNCTIONPREFIX1,$FUNCTIONNAME)]|document($DOXYGENDOC)/doxygen/compounddef/compoundname[.=concat($DOXYGENCLASSPREFIX,$CLASSNAME)]/../sectiondef/memberdef/name[.=concat($DOXYGENFUNCTIONPREFIX2,$FUNCTIONNAME)]"/>
        </xs:annotation>
      </xsl:if>
      <xsl:apply-templates>
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        <xsl:with-param name="FUNCTIONNAME" select="$FUNCTIONNAME"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

  <!-- append doxygen to existing xs:annotation -->
  <xsl:template match="xs:element/xs:annotation" priority="1">
    <xsl:param name="CLASSNAME"/>
    <xsl:param name="FUNCTIONNAME"/>
    <xsl:copy>
      <xsl:apply-templates/>
      <xsl:apply-templates select="document($DOXYGENDOC)/doxygen/compounddef/compoundname[.=concat($DOXYGENCLASSPREFIX,$CLASSNAME)]/../sectiondef/memberdef/name[.=concat($DOXYGENFUNCTIONPREFIX1,$FUNCTIONNAME)]|document($DOXYGENDOC)/doxygen/compounddef/compoundname[.=concat($DOXYGENCLASSPREFIX,$CLASSNAME)]/../sectiondef/memberdef/name[.=concat($DOXYGENFUNCTIONPREFIX2,$FUNCTIONNAME)]"/>
    </xsl:copy>
  </xsl:template>

  <!-- add doxygen class member documentation -->
  <xsl:template match="/doxygen/compounddef/sectiondef/memberdef/name">
    <xsl:if test="string-length(normalize-space(../briefdescription))!=0 or string-length(normalize-space(../detaileddescription))!=0">
      <xs:documentation source="doxygen">
        <xsl:if test="string-length(normalize-space(../briefdescription))!=0">
          <xsl:apply-templates select="../briefdescription"/>
        </xsl:if>
        <xsl:if test="string-length(normalize-space(../detaileddescription))!=0">
          <xsl:apply-templates select="../detaileddescription"/>
        </xsl:if>
      </xs:documentation>
    </xsl:if>
  </xsl:template>

</xsl:stylesheet>
