# Note: with automake 1.14 we can use %reldir% instead of $(prefix)/share/mbxmlutils/python!
deplibs.target:
	set -e; \
	echo "Create dependency file(s) for $(lib_LTLIBRARIES)"
	FILES=""; \
	for lib_la in $(lib_LTLIBRARIES); do \
	  . $(libdir)/$$lib_la; \
	  if test -n "$$dlname"; then \
	    FILES="$$FILES $$libdir/$$dlname"; \
	  fi; \
	done; \
	python3 $(prefix)/share/mbxmlutils/python/deplibs.py -b $$FILES
