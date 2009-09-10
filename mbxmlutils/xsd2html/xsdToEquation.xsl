<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:html="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- output method -->
  <xsl:output method="text"/>

  <xsl:template match="text()"/>

  <xsl:template match="@*"/>

  <xsl:template match="html:img[@class='eqn']"><xsl:text>MBXMLUTILS_BEGINEQN
</xsl:text><xsl:value-of select="@src"/><xsl:text>
\[ </xsl:text><xsl:value-of select="@alt"/><xsl:text> \]
MBXMLUTILS_ENDEQN
</xsl:text></xsl:template>

  <xsl:template match="html:img[@class='inlineeqn']"><xsl:text>MBXMLUTILS_BEGINEQN
</xsl:text><xsl:value-of select="@src"/><xsl:text>
$ </xsl:text><xsl:value-of select="@alt"/><xsl:text> $
MBXMLUTILS_ENDEQN
</xsl:text></xsl:template>
 
</xsl:stylesheet>
