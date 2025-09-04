# USAGE
# install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -DLIBS=${CMAKE_INSTALL_PREFIX}/$<IF:$<PLATFORM_ID:Windows>,${CMAKE_INSTALL_BINDIR},${CMAKE_INSTALL_LIBDIR}>/$<TARGET_FILE_NAME:mbsimRCStengel> -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -P ${CMAKE_INSTALL_PREFIX}/share/mbxmlutils/python/deplibs.cmake)")
#
# NOTE
# comma ',' is used as seperator in -DLIBS to avoid that ';' is interpreted by the shell

string(REPLACE "," " " LIBS ${LIBS})
execute_process(COMMAND python3 ${CMAKE_INSTALL_PREFIX}/share/mbxmlutils/python/deplibs.py -b ${LIBS})
