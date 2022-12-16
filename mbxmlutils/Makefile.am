SUBDIRS = mbxmlutilshelper mbxmlutils xsd2html
DIST_SUBDIRS = mbxmlutilshelper mbxmlutils xsd2html xmldoc doc

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = mbxmlutils.pc mbxmlutilshelper.pc

licdir = @datadir@/mbxmlutils
lic_DATA = COPYING NOTICE



# doxygen
.PHONY: doc doc_install doc_uninstall
doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) all)
doc_install:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) install)
doc_uninstall:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall)

# create clang compile database if compiledb command is found
all-local: compile_commands.json
check-local: compile_commands.json
install-data-local: compile_commands.json
install-exec-local: compile_commands.json
compile_commands.json: config.log
	command -v compiledb &> /dev/null && compiledb -n -o compile_commands.json make && ln -fs $(abs_top_builddir)/compile_commands.json $(top_srcdir)/compile_commands.json || echo "Failed to create clang compile DB"
