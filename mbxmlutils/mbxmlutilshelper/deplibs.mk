# Note: with automake 1.14 we can use %reldir% instead of $(prefix)/share/mbxmlutils/python!
deplibs.target:
	set -e; \
	echo "Create dependency file(s) (only if python is found) for $(lib_LTLIBRARIES)"
	if which python &> /dev/null; then \
	  for lib_la in $(lib_LTLIBRARIES); do \
	    . $(libdir)/$$lib_la; \
	    if test -n "$$dlname"; then \
	      if test $$libdir/$$dlname -nt $$libdir/$$dlname.deplibs; then \
	        echo "Create dependency files for $$libdir/$$dlname"; \
	        python3 $(prefix)/share/mbxmlutils/python/deplibs.py $$libdir/$$dlname > $$libdir/$$dlname.deplibs.tmp; \
	        mv -f $$libdir/$$dlname.deplibs.tmp $$libdir/$$dlname.deplibs; \
	      fi; \
	    fi; \
	  done; \
	fi
