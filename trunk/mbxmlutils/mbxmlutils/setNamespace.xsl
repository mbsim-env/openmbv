<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <xsl:param name="NAMESPACE"/>
  <xsl:param name="NAMESPACELOCATION"/>

  <!-- clone -->

  <xsl:template match="*">
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <xsl:apply-templates/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="text()|comment()|processing-instruction()">
    <xsl:copy/>
  </xsl:template>

  <xsl:template match="@*">
    <xsl:attribute name="{name(.)}">
      <xsl:value-of select="replace(replace(.,'@NAMESPACE@',$NAMESPACE),'@NAMESPACELOCATION@',$NAMESPACELOCATION)"/>
    </xsl:attribute>
  </xsl:template>

</xsl:stylesheet>
