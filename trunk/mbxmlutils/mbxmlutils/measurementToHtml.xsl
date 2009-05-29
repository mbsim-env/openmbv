<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://openmbv.berlios.de/MBXMLUtils/measurement"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- output method -->
  <xsl:output method="html"
    doctype-public="-//W3C//DTD XHTML 1.0 Strict//EN"
    doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <!-- for all direct root child elements (classes) -->
  <xsl:template match="/">
    <!-- html header -->
    <html xml:lang="en" lang="en"><head><title>Measurement XML Documentation</title></head><body>
    <h1>Measurement XML Documentation</h1>
    <p>This is the documentation of the definition of scalar, vector and matrix types with a abitary measurement.</p>
    <h2>Contents</h2>
    <ul>
      <li><a name="content-scalartype" href="#scalartype">Scalar Type</a></li>
      <li><a name="content-vectortype" href="#vectortype">Vector Type</a></li>
      <li><a name="content-matrixtype" href="#matrixtype">Matrix Type</a></li>
      <li><a name="content-measurements" href="#measurements">Measurements</a>
        <ul>
          <xsl:for-each select="/mm:measurement/mm:measure">
            <xsl:sort select="@name"/>
            <li>
              <a><xsl:attribute name="name">content-<xsl:value-of select="@name"/></xsl:attribute>
                <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute><xsl:value-of select="@name"/></a>
            </li>
          </xsl:for-each>
        </ul>
      </li>
    </ul>

    <h2><a name="scalartype" href="#content-scalartype">Scalar Type</a></h2>
    <p>A scalar type can be of any unit defined in <a href="#measurements">measurements</a>.
      The type name of a scalar of measure length is <span style="font-family:monospace">pv:lengthScalar</span> and so on.
      Where <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span>.</p>
    <p>The content of a scalar type must be a text value. All parameters defined in the parameter file given to the XML preprocessor are substututed in this text value. Afterwards this text value, which can be a abitary octave  string/program, is evaluated by octave. The text value must evalute to a single scalar value. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
    <pre>&lt;myScalarElement&gt;4*sin(a)+b&lt;/myScalarElement&gt;</pre>
    <pre>&lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;</pre></p>

    <h2><a name="vectortype" href="#content-vectortype">Vector Type</a></h2>
    <p>A vector type can be of any unit defined in <a href="#measurements">measurements</a>.
      The type name of a vector of measure length is <span style="font-family:monospace">pv:lengthVector</span> and so on.
      Where <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span>.</p>
    <p>The content of a vector type can be one of the following:
      <ul>
        <li>A text value: All parameters defined in the parameter file given to the XML preprocessor are substututed in this text value. Afterwards this text value, which can be a abitary octave  string/program, is evaluated by octave. The text value must evalute to a single row vector value. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
            <pre>&lt;myVectorElement&gt;[1;b;a;7]&lt;/myVectorElement&gt;</pre>
            <pre>&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;</pre></li>
        <li>A XML representation of a vector: The following shows a example of such a XML representation.<pre>
&lt;myVectorElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;7.3&lt;/pv:ele&gt;
&lt;/myVectorElement&gt;
</pre></li>
        <li>A reference to a ascii file: The following shows a example as a reference to the file vec.txt.<pre>
&lt;myVectorElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiVectorRef href="vec.txt"/&gt;
&lt;/myVectorElement&gt;
</pre>The file vec.txt is a simple ascii file containing one element of the vector per line. All empty lines are ignored and the the content between '#' or '%' and the end of line is also ignored (comments).</li>
      </ul>
    </p>

    <h2><a name="matrixtype" href="#content-matrixtype">Matrix Type</a></h2>
    <p>A matrix type can be of any unit defined in <a href="#measurements">measurements</a>.
      The type name of a matrix of measure length is <span style="font-family:monospace">pv:lengthMatrix</span> and so on.
      Where <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span>.</p>
    <p>The content of a matrix type can be one of the following:
      <ul>
        <li>A text value: All parameters defined in the parameter file given to the XML preprocessor are substututed in this text value. Afterwards this text value, which can be a abitary octave  string/program, is evaluated by octave. The text value must evalute to a single matrix value. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
            <pre>&lt;myMatrixElement&gt;[1,b;a,7]&lt;/myMatrixElement&gt;</pre>
            <pre>&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;</pre></li>
        <li>A XML representation of a matrix: The following shows a example of such a XML representation.<pre>
&lt;myMatrixElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
&lt;/myMatrixElement&gt;
</pre></li>
        <li>A reference to a ascii file: The following shows a example as a reference to the file mat.txt.<pre>
&lt;myMatrixElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiMatrixRef href="mat.txt"/&gt;
&lt;/myMatrixElement&gt;
</pre>The file mat.txt is a simple ascii file containing one row of the vector per line. The values inside a row must be separated by ',' or space. All empty lines are ignored and the the content between '#' or '%' and the end of line is also ignored (comments).</li>
      </ul>
    </p>

    <h2><a name="measurements" href="#content-measurements">Measurements</a></h2>
    <p>The following measurements are defined</p>
    <xsl:apply-templates select="/mm:measurement/mm:measure">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    </body></html>
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
    <h3><a>
      <xsl:attribute name="name">
        <xsl:value-of select="@name"/>
      </xsl:attribute>
      <xsl:attribute name="href">#content-<xsl:value-of select="@name"/></xsl:attribute>
      <xsl:value-of select="@name"/>
    </a></h3>
    <p>The SI unit of <xsl:value-of select="@name"/> is: <span style="font-weight:bold"><xsl:value-of select="@SIunit"/></span></p>
    <p>The following units are defined the measure <xsl:value-of select="@name"/>. "Unit Name" is the name of the unit and
      "Conversion to SI Unit" is a expression which converts a value of this unit to the SI unit.</p>
    <table border="1">
      <thead> <tr> <th>Unit Name</th> <th>Conversion to SI Unit</th> </tr> </thead>
      <xsl:apply-templates select="mm:unit"/>
    </table>
  </xsl:template>

  <xsl:template match="mm:unit">
    <tr>
      <!-- if SI unit use bold font -->
      <xsl:if test="../@SIunit=@name">
        <td><span style="font-weight:bold"><xsl:value-of select="@name"/></span></td>
      </xsl:if>
      <!-- if not SI unit use normalt font -->
      <xsl:if test="../@SIunit!=@name">
        <td><xsl:value-of select="@name"/></td>
      </xsl:if>
      <!-- outout conversion -->
      <td><xsl:value-of select="."/></td>
    </tr>
  </xsl:template>

</xsl:stylesheet>
