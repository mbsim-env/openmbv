# USAGE
# install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -DLIBS=${CMAKE_INSTALL_PREFIX}/lib/$<TARGET_FILE_PREFIX:mytarget>$<TARGET_FILE_BASE_NAME:mytarget>$<TARGET_FILE_SUFFIX:mytarget> -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -P ${CMAKE_INSTALL_PREFIX}/share/mbxmlutils/python/deplibs.cmake)")
#
# NOTE
# comma ',' is used as seperator in -DLIBS to avoid that ';' is interpreted by the shell

string(REPLACE "," " " LIBS ${LIBS})
execute_process(COMMAND python3 ${CMAKE_INSTALL_PREFIX}/share/mbxmlutils/python/deplibs.py -b ${LIBS})
