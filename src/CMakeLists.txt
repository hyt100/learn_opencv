add_executable(001 001_version.cc)
target_link_libraries(001 ${OpenCV_LIBS})

add_executable(002 002_rotate.cc)
target_link_libraries(002 ${OpenCV_LIBS})

add_executable(003 003_photocorrect.cc)
target_link_libraries(003 ${OpenCV_LIBS})

add_executable(004 004_feathering.cc)
target_link_libraries(004 ${OpenCV_LIBS})

add_executable(005 005_colorspace_convert.cc)
target_link_libraries(005 ${OpenCV_LIBS})

add_executable(006 006_hist.cc)
target_link_libraries(006 ${OpenCV_LIBS})

add_executable(007 007_resize.cc)
target_link_libraries(007 ${OpenCV_LIBS})

add_executable(008 008_adjust.cc)
target_link_libraries(008 ${OpenCV_LIBS})