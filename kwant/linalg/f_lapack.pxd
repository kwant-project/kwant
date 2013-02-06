# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

ctypedef int l_int
ctypedef int l_logical

cdef extern:
  void sgetrf_(l_int *, l_int *, float *, l_int *, l_int *, l_int *)
  void dgetrf_(l_int *, l_int *, double *, l_int *, l_int *, l_int *)
  void cgetrf_(l_int *, l_int *, float complex *, l_int *, l_int *,
               l_int *)
  void zgetrf_(l_int *, l_int *, double complex *, l_int *, l_int *,
               l_int *)

  void sgetrs_(char *, l_int *, l_int *, float *, l_int *, l_int *,
               float *, l_int *, l_int *)
  void dgetrs_(char *, l_int *, l_int *, double *, l_int *, l_int *,
               double *, l_int *, l_int *)
  void cgetrs_(char *, l_int *, l_int *, float complex *, l_int *,
               l_int *, float complex *, l_int *, l_int *)
  void zgetrs_(char *, l_int *, l_int *, double complex *, l_int *,
               l_int *, double complex *, l_int *, l_int *)

  void sgecon_(char *, l_int *, float *, l_int *, float *, float *,
               float *, l_int *, l_int *)
  void dgecon_(char *, l_int *, double *, l_int *, double *, double *,
               double *, l_int *, l_int *)
  void cgecon_(char *, l_int *, float complex *, l_int *, float *,
               float *, float complex *, float *, l_int *)
  void zgecon_(char *, l_int *, double complex *, l_int *, double *,
               double *, double complex *, double *, l_int *)

  void sggev_(char *, char *, l_int *, float *, l_int *, float *, l_int *,
              float *, float *, float *, float *, l_int *, float *, l_int *,
              float *, l_int *, l_int *)
  void dggev_(char *, char *, l_int *, double *, l_int *, double *, l_int *,
              double *, double *, double *, double *, l_int *,
              double *, l_int *, double *, l_int *, l_int *)
  void cggev_(char *, char *, l_int *, float complex *, l_int *,
              float complex *, l_int *, float complex *, float complex *,
              float complex *, l_int *, float complex *, l_int *,
              float complex *, l_int *, float *, l_int *)
  void zggev_(char *, char *, l_int *, double complex *, l_int *,
              double complex *, l_int *, double complex *,
              double complex *, double complex *, l_int *,
              double complex *, l_int *, double complex *, l_int *,
              double *, l_int *)

  void sgees_(char *, char *, l_logical (*)(float *, float *),
              l_int *, float *, l_int *, l_int *,
              float *, float *, float *, l_int *,
              float *, l_int *, l_logical *, l_int *)
  void dgees_(char *, char *, l_logical (*)(double *, double *),
              l_int *, double *, l_int *, l_int *,
              double *, double *, double *, l_int *,
              double *, l_int *, l_logical *, l_int *)
  void cgees_(char *, char *,
              l_logical (*)(float complex *),
              l_int *, float complex *,
              l_int *, l_int *, float complex *,
              float complex *, l_int *,
              float complex *, l_int *, float *,
              l_logical *, l_int *)
  void zgees_(char *, char *,
              l_logical (*)(double complex *),
              l_int *, double complex *,
              l_int *, l_int *, double complex *,
              double complex *, l_int *,
              double complex *, l_int *,
              double *, l_logical *, l_int *)

  void strsen_(char *, char *, l_logical *, l_int *,
               float *, l_int *, float *,
               l_int *, float *, float *, l_int *,
               float *, float *, float *, l_int *,
               l_int *, l_int *, l_int *)
  void dtrsen_(char *, char *, l_logical *,
               l_int *, double *, l_int *,
               double *, l_int *, double *, double *,
               l_int *, double *, double *, double *,
               l_int *, l_int *, l_int *, l_int *)
  void ctrsen_(char *, char *, l_logical *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, float complex *, l_int *,
               float *, float *, float complex *,
               l_int *, l_int *)
  void ztrsen_(char *, char *, l_logical *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, double complex *, l_int *,
               double *, double *, double complex *,
               l_int *, l_int *)

  void strevc_(char *, char *, l_logical *,
               l_int *, float *, l_int *,
               float *, l_int *, float *, l_int *,
               l_int *, l_int *, float *, l_int *)
  void dtrevc_(char *, char *, l_logical *,
               l_int *, double *, l_int *,
               double *, l_int *, double *,
               l_int *, l_int *, l_int *, double *,
               l_int *)
  void ctrevc_(char *, char *, l_logical *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, l_int *, l_int *,
               float complex *, float *, l_int *)
  void ztrevc_(char *, char *, l_logical *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, l_int *, l_int *,
               double complex *, double *, l_int *)

  void sgges_(char *, char *, char *,
              l_logical (*)(float *, float *, float *),
              l_int *, float *, l_int *, float *,
              l_int *, l_int *, float *, float *,
              float *, float *, l_int *, float *,
              l_int *, float *, l_int *, l_logical *,
              l_int *)
  void dgges_(char *, char *, char *,
              l_logical (*)(double *, double *, double *),
              l_int *, double *, l_int *, double *,
              l_int *, l_int *, double *, double *,
              double *, double *, l_int *, double *,
              l_int *, double *, l_int *,
              l_logical *, l_int *)
  void cgges_(char *, char *, char *,
              l_logical (*)(float complex *, float complex *),
              l_int *, float complex *,
              l_int *, float complex *,
              l_int *, l_int *, float complex *,
              float complex *, float complex *,
              l_int *, float complex *,
              l_int *, float complex *,
              l_int *, float *, l_logical *, l_int *)
  void zgges_(char *, char *, char *,
              l_logical (*)(double complex *, double complex *),
              l_int *, double complex *,
              l_int *, double complex *,
              l_int *, l_int *, double complex *,
              double complex *,
              double complex *, l_int *,
              double complex *, l_int *,
              double complex *, l_int *,
              double *, l_logical *, l_int *)

  void stgsen_(l_int *, l_logical *,
               l_logical *, l_logical *,
               l_int *, float *, l_int *, float *,
               l_int *, float *, float *, float *,
               float *, l_int *, float *, l_int *,
               l_int *, float *, float *, float *, float *,
               l_int *, l_int *, l_int *, l_int *)
  void dtgsen_(l_int *, l_logical *,
               l_logical *, l_logical *,
               l_int *, double *, l_int *,
               double *, l_int *, double *, double *,
               double *, double *, l_int *, double *,
               l_int *, l_int *, double *, double *,
               double *, double *, l_int *, l_int *,
               l_int *, l_int *)
  void ctgsen_(l_int *, l_logical *,
               l_logical *, l_logical *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, float complex *,
               float complex *,
               float complex *, l_int *,
               float complex *, l_int *, l_int *,
               float *, float *, float *,
               float complex *, l_int *, l_int *,
               l_int *, l_int *)
  void ztgsen_(l_int *, l_logical *,
               l_logical *, l_logical *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, double complex *,
               double complex *,
               double complex *, l_int *,
               double complex *, l_int *, l_int *,
               double *, double *, double *,
               double complex *, l_int *, l_int *,
               l_int *, l_int *)

  void stgevc_(char *, char *, l_logical *,
               l_int *, float *, l_int *,
               float *, l_int *, float *,
               l_int *, float *, l_int *,
               l_int *, l_int *, float *, l_int *)
  void dtgevc_(char *, char *, l_logical *,
               l_int *, double *, l_int *,
               double *, l_int *, double *,
               l_int *, double *, l_int *,
               l_int *, l_int *, double *, l_int *)
  void ctgevc_(char *, char *, l_logical *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, float complex *,
               l_int *, l_int *, l_int *,
               float complex *, float *, l_int *)
  void ztgevc_(char *, char *, l_logical *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, double complex *,
               l_int *, l_int *, l_int *,
               double complex *, double *, l_int *)
