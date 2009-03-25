dnl -*- mode: autoconf -*-

AC_DEFUN([ES_CHECK_FFTW],[
	AC_ARG_WITH([fftw],
		AC_HELP_STRING([--with-fftw=VERSION],
			[specify the version of FFTW to use (2 or 3)]),
		, with_fftw=guess)
	dnl with_fftw=no    don't use FFTW
	dnl with_fftw=yes   try to find a working FFTW, bail out if none is found
	dnl with_fftw=  (not set) try to find a working FFTW, continue if none is found
	dnl otherwise       use the specified version

        LIBS=" $LIBS -lm "

	if test x$with_fftw = xguess || test x$with_fftw = xyes; then
     		# search for FFTW
		ES_CHECK_FFTW3
		if test x$fftw3_found = xyes; then
		   use_fftw=3
		else
		  ES_CHECK_FFTW2
		  if test x$fftw2_found = xyes; then
		     use_fftw=2
		  elif test x$with_fftw = xyes; then
		       ES_NOTE_64BIT
		       AC_MSG_FAILURE([
********************************************************************************
* Could not find FFTW!                                                         *
********************************************************************************
])
		  fi
		fi
	elif test x$with_fftw = x3; then
                use_fftw=3
		ES_CHECK_FFTW3
		if test x$fftw3_found != xyes; then
		   ES_NOTE_64BIT
		   AC_MSG_FAILURE([
********************************************************************************
* Could not find FFTW3!                                                        *
* Please add the library path to LDFLAGS (e.g. configure LDFLAGS=-L/usr/lib)   *
* Please add the include path to CPPFLAGS                                      *
* (e.g. configure CPPFLAGS=-I/usr/include).                                    *
********************************************************************************
])
		fi
	elif test x$with_fftw = x2; then
		use_fftw=2
		ES_CHECK_FFTW2
		if test x$fftw2_found != xyes; then
		   ES_NOTE_64BIT
		   AC_MSG_FAILURE([
********************************************************************************
* Could not find FFTW2!                                                        *
* Please add the library path to LDFLAGS (e.g. configure LDFLAGS=-L/usr/lib)   *
* Please add the include path to CPPFLAGS                                      *
* (e.g. configure CPPFLAGS=-I/usr/include).                                    *
********************************************************************************
])
		fi
	elif test x$with_fftw != xno; then
	  AC_MSG_ERROR([specified bad FFTW version ($with_fftw)])
	fi

	# now save the result
	if test x$use_fftw = xguess; then
	   use_fftw=none
	else
	   AC_DEFINE_UNQUOTED(FFTW, $use_fftw, [Whether to use the FFTW library, and which version to use])
	fi
])

AC_DEFUN([ES_CHECK_FFTW3],[
 	ES_ADDPATH_CHECK_LIB(fftw3, fftw_plan_many_dft, [fftw3_found=yes], [fftw3_found=no])
	if test x$fftw3_found = xyes; then
		ES_ADDPATH_CHECK_HEADER(fftw3.h, [], [fftw3_found=no])
	fi
])

AC_DEFUN([ES_CHECK_FFTW2],[
	dnl we just assume that fftw and rfftw are in the same directory
	dnl if this is not the case for you, consider cleaning up your system...
	dnl first we check for the d* (SuSE) versions, then the normal ones
	dnl At the end we have to include rfftw before fftw on some systems, but testing
	dnl is only possible for fftw
	saved_LIBS=$LIBS
 	ES_ADDPATH_CHECK_LIB(dfftw, fftw_create_plan_specific, [fftw2_found=yes], [fftw2_found=no])
	if test x$fftw2_found = xyes; then
		LIBS="$saved_LIBS -ldrfftw -ldfftw"
	else
	 	ES_ADDPATH_CHECK_LIB(fftw, fftw_create_plan_specific, [fftw2_found=yes], [fftw2_found=no])
		if test x$fftw2_found = xyes; then
			LIBS="$LIBS -lrfftw -lfftw"
		fi
	fi
	if test x$fftw2_found = xyes; then
		ES_ADDPATH_CHECK_HEADER(fftw.h, [], [fftw2_found=no])
	fi
])

