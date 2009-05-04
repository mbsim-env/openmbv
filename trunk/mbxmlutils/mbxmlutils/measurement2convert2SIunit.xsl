<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xslo="http://www.w3.org/1999/XSL/TransformOUTPUT"
  xmlns:mm="http://openmbv.berlios.de/MBXMLUtils/measurement">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <xsl:param name="XMLDIR"/>

  <xsl:namespace-alias stylesheet-prefix="xslo" result-prefix="xsl"/>

  <xsl:template match="/">
    <xsl:comment>
      <!-- Add this text to the output document to prevent others editing the output document manual -->
      This file is generated automatically using XSLT from measurement.xml.
      DO NOT EDIT!!!
    </xsl:comment>

    <xslo:stylesheet version="2.0"
      xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable">

    
      <xslo:output method="xml" version="1.0" indent="yes"/>

      <xsl:comment>clone</xsl:comment>
    
      <xslo:template match="*">
        <xslo:copy>
          <xslo:apply-templates select="@*"/>
          <xslo:apply-templates/>
        </xslo:copy>
      </xslo:template>
    
      <xslo:template match="@*">
        <xslo:copy/>
      </xslo:template>
    
    
    
      <xsl:comment>deunit</xsl:comment>
    
      <xslo:import-schema namespace="@NAMESPACE@" schema-location="@NAMESPACELOCATION@"/>
    
      <xsl:apply-templates select="document('measurement.xml')/mm:measurement/*"/>
    
      <xslo:template mode="UNIT" match="/mm:measurement/mm:measure/mm:unit">
        <xslo:param name="CURUNIT"/>
        <xslo:param name="CURVALUE"/>
        <xslo:if test="@name=$CURUNIT">
          <xslo:value-of select="replace(.,'value',concat('(',normalize-space($CURVALUE),')'))"/>
        </xslo:if>
      </xslo:template>
    
    </xslo:stylesheet>
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
    <xslo:template>
      <xsl:attribute name="match">element(*,pv:<xsl:value-of select="@name"/>Scalar)|element(*,pv:<xsl:value-of select="@name"/>Vector)|element(*,pv:<xsl:value-of select="@name"/>Matrix)</xsl:attribute>
      <xslo:copy>
        <xslo:apply-templates mode="UNIT">
          <xsl:attribute name="select">document('<xsl:value-of select="$XMLDIR"/>/measurement.xml')/mm:measurement/mm:measure[@name='<xsl:value-of select="@name"/>']/mm:unit</xsl:attribute>
          <xslo:with-param name="CURUNIT" select="@unit"/>
          <xslo:with-param name="CURVALUE" select="."/>
        </xslo:apply-templates>
      </xslo:copy>
    </xslo:template>
  </xsl:template>

</xsl:stylesheet>
