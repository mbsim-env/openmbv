# Note: with automake 1.14 we can use %reldir% instead of $(prefix)/share/mbxmlutils/python!
deplibs.target:
	set -e; \
	echo "Create dependency file(s) (only if python3 is found) for $(lib_LTLIBRARIES)"
	if which python3 &> /dev/null; then \
	  ANYFAILED=0; \
	  for lib_la in $(lib_LTLIBRARIES); do \
	    . $(libdir)/$$lib_la; \
	    if test -n "$$dlname"; then \
	      if test $$libdir/$$dlname -nt $$libdir/$$dlname.deplibs; then \
	        echo "Create dependency files for $$libdir/$$dlname"; \
	        python3 $(prefix)/share/mbxmlutils/python/deplibs.py $$libdir/$$dlname > $$libdir/$$dlname.deplibs.tmp || ANYFAILED=1; \
	        mv -f $$libdir/$$dlname.deplibs.tmp $$libdir/$$dlname.deplibs; \
	      fi; \
	    fi; \
	  done; \
	  if test $$ANYFAILED = 1; then \
	    echo "At least one deplibs command failed, see above."; \
	    exit 1; \
	  fi; \
	fi
