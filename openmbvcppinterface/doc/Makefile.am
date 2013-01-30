docdir = @datadir@/doc/@PACKAGE@

clean:
	rm -rf html

all:
	$(doxygen)

install:
	$(MKDIR_P) $(docdir); \
	cp -r html $(docdir); \
	cp $(PACKAGE).tag $(docdir)

uninstall:
	rm -rf $(docdir)/html
	rm -rf $(docdir)/$(PACKAGE).tag
