# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from libc.stdio cimport FILE

cdef extern from "scotch.h":
    ctypedef struct SCOTCH_Arch:
        pass

    ctypedef struct SCOTCH_Geom:
        pass

    ctypedef struct SCOTCH_Graph:
        double dummy[12]

    ctypedef struct SCOTCH_Mesh:
        pass

    ctypedef struct SCOTCH_Mapping:
        pass

    ctypedef struct SCOTCH_Ordering:
        pass

    ctypedef struct SCOTCH_Strat:
        pass

    ctypedef int SCOTCH_Idx

    ctypedef int SCOTCH_Num

    void SCOTCH_errorProg(char *)
    void SCOTCH_errorPrint(char *, ...)
    void SCOTCH_errorPrintW(char *, ...)

    int SCOTCH_archInit(SCOTCH_Arch *)
    void SCOTCH_archExit(SCOTCH_Arch *)
    int SCOTCH_archLoad(SCOTCH_Arch *, FILE *)
    int SCOTCH_archSave(SCOTCH_Arch *, FILE *)
    int SCOTCH_archBuild(SCOTCH_Arch *, SCOTCH_Graph *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Strat *)
    char *SCOTCH_archName(SCOTCH_Arch *)
    SCOTCH_Num SCOTCH_archSize(SCOTCH_Arch *)
    int SCOTCH_archCmplt(SCOTCH_Arch *, SCOTCH_Num)
    int SCOTCH_archCmpltw(SCOTCH_Arch *, SCOTCH_Num, SCOTCH_Num *)
    int SCOTCH_archHcub(SCOTCH_Arch *, SCOTCH_Num)
    int SCOTCH_archMesh2(SCOTCH_Arch *, SCOTCH_Num, SCOTCH_Num)
    int SCOTCH_archMesh3(SCOTCH_Arch *, SCOTCH_Num, SCOTCH_Num, SCOTCH_Num)
    int SCOTCH_archTleaf(SCOTCH_Arch *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Num *)
    int SCOTCH_archTorus2(SCOTCH_Arch *, SCOTCH_Num, SCOTCH_Num)
    int SCOTCH_archTorus3(SCOTCH_Arch *, SCOTCH_Num, SCOTCH_Num, SCOTCH_Num)
    int SCOTCH_archVcmplt(SCOTCH_Arch *)
    int SCOTCH_archVhcub(SCOTCH_Arch *)

    int SCOTCH_geomInit(SCOTCH_Geom *)
    void SCOTCH_geomExit(SCOTCH_Geom *)
    void SCOTCH_geomData(SCOTCH_Geom *, SCOTCH_Num *, double **)

    int SCOTCH_graphInit(SCOTCH_Graph *)
    void SCOTCH_graphExit(SCOTCH_Graph *)
    void SCOTCH_graphFree(SCOTCH_Graph *)
    int SCOTCH_graphLoad(SCOTCH_Graph *, FILE *, SCOTCH_Num, SCOTCH_Num)
    int SCOTCH_graphSave(SCOTCH_Graph *, FILE *)
    int SCOTCH_graphBuild(SCOTCH_Graph *, SCOTCH_Num, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Num *)
    SCOTCH_Num SCOTCH_graphBase(SCOTCH_Graph *, SCOTCH_Num baseval)
    int SCOTCH_graphCheck(SCOTCH_Graph *)
    void SCOTCH_graphSize(SCOTCH_Graph *, SCOTCH_Num *, SCOTCH_Num *)
    void SCOTCH_graphData(SCOTCH_Graph *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num *, SCOTCH_Num **, SCOTCH_Num **)
    void SCOTCH_graphStat(SCOTCH_Graph *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, double *, double *, SCOTCH_Num *, SCOTCH_Num *, double *, double *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, double *, double *)
    int SCOTCH_graphGeomLoadChac(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_graphGeomLoadHabo(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_graphGeomLoadMmkt(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_graphGeomLoadScot(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_graphGeomSaveChac(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_graphGeomSaveMmkt(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_graphGeomSaveScot(SCOTCH_Graph *, SCOTCH_Geom *, FILE *, FILE *, char *)

    int SCOTCH_graphMapInit(SCOTCH_Graph *, SCOTCH_Mapping *, SCOTCH_Arch *, SCOTCH_Num *)
    void SCOTCH_graphMapExit(SCOTCH_Graph *, SCOTCH_Mapping *)
    int SCOTCH_graphMapLoad(SCOTCH_Graph *, SCOTCH_Mapping *, FILE *)
    int SCOTCH_graphMapSave(SCOTCH_Graph *, SCOTCH_Mapping *, FILE *)
    int SCOTCH_graphMapView(SCOTCH_Graph *, SCOTCH_Mapping *, FILE *)
    int SCOTCH_graphMapCompute(SCOTCH_Graph *, SCOTCH_Mapping *, SCOTCH_Strat *)
    int SCOTCH_graphMap(SCOTCH_Graph *, SCOTCH_Arch *, SCOTCH_Strat *, SCOTCH_Num *)
    int SCOTCH_graphPart(SCOTCH_Graph *, SCOTCH_Num, SCOTCH_Strat *, SCOTCH_Num *)

    int SCOTCH_graphOrderInit(SCOTCH_Graph *, SCOTCH_Ordering *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    void SCOTCH_graphOrderExit(SCOTCH_Graph *, SCOTCH_Ordering *)
    int SCOTCH_graphOrderLoad(SCOTCH_Graph *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_graphOrderSave(SCOTCH_Graph *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_graphOrderSaveMap(SCOTCH_Graph *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_graphOrderSaveTree(SCOTCH_Graph *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_graphOrderCompute(SCOTCH_Graph *, SCOTCH_Ordering *, SCOTCH_Strat *)
    int SCOTCH_graphOrderComputeList(SCOTCH_Graph *, SCOTCH_Ordering *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Strat *)
    int SCOTCH_graphOrderFactor(SCOTCH_Graph *, SCOTCH_Ordering *, SCOTCH_Graph *)
    int SCOTCH_graphOrderView(SCOTCH_Graph *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_graphOrder(SCOTCH_Graph *, SCOTCH_Strat *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    int SCOTCH_graphOrderList(SCOTCH_Graph *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Strat *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    int SCOTCH_graphOrderCheck(SCOTCH_Graph *, SCOTCH_Ordering *)

    int SCOTCH_meshInit(SCOTCH_Mesh *)
    void SCOTCH_meshExit(SCOTCH_Mesh *)
    int SCOTCH_meshLoad(SCOTCH_Mesh *, FILE *, SCOTCH_Num)
    int SCOTCH_meshSave(SCOTCH_Mesh *, FILE *)
    int SCOTCH_meshBuild(SCOTCH_Mesh *, SCOTCH_Num, SCOTCH_Num, SCOTCH_Num, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num, SCOTCH_Num *)
    int SCOTCH_meshCheck(SCOTCH_Mesh *)
    void SCOTCH_meshSize(SCOTCH_Mesh *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    void SCOTCH_meshData(SCOTCH_Mesh *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num **, SCOTCH_Num *, SCOTCH_Num **, SCOTCH_Num *)
    void SCOTCH_meshStat(SCOTCH_Mesh *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, double *, double *, SCOTCH_Num *, SCOTCH_Num *, double *, double *, SCOTCH_Num *, SCOTCH_Num *, double *, double *)
    int SCOTCH_meshGraph(SCOTCH_Mesh *, SCOTCH_Graph *)
    int SCOTCH_meshGeomLoadHabo(SCOTCH_Mesh *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_meshGeomLoadScot(SCOTCH_Mesh *, SCOTCH_Geom *, FILE *, FILE *, char *)
    int SCOTCH_meshGeomSaveScot(SCOTCH_Mesh *, SCOTCH_Geom *, FILE *, FILE *, char *)

    int SCOTCH_meshOrderInit(SCOTCH_Mesh *, SCOTCH_Ordering *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    void SCOTCH_meshOrderExit(SCOTCH_Mesh *, SCOTCH_Ordering *)
    int SCOTCH_meshOrderSave(SCOTCH_Mesh *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_meshOrderSaveMap(SCOTCH_Mesh *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_meshOrderSaveTree(SCOTCH_Mesh *, SCOTCH_Ordering *, FILE *)
    int SCOTCH_meshOrderCompute(SCOTCH_Mesh *, SCOTCH_Ordering *, SCOTCH_Strat *)
    int SCOTCH_meshOrderComputeList(SCOTCH_Mesh *, SCOTCH_Ordering *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Strat *)
    int SCOTCH_meshOrder(SCOTCH_Mesh *, SCOTCH_Strat *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    int SCOTCH_meshOrderList(SCOTCH_Mesh *, SCOTCH_Num, SCOTCH_Num *, SCOTCH_Strat *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *, SCOTCH_Num *)
    int SCOTCH_meshOrderCheck(SCOTCH_Mesh *, SCOTCH_Ordering *)

    void SCOTCH_randomReset()

    int SCOTCH_stratInit(SCOTCH_Strat *)
    void SCOTCH_stratExit(SCOTCH_Strat *)
    void SCOTCH_stratFree(SCOTCH_Strat *)
    int SCOTCH_stratSave(SCOTCH_Strat *, FILE *)

    int SCOTCH_stratGraphBipart(SCOTCH_Strat *, char *)
    int SCOTCH_stratGraphMap(SCOTCH_Strat *, char *)
    int SCOTCH_stratGraphMapBuild(SCOTCH_Strat *, SCOTCH_Num, SCOTCH_Num, double)
    int SCOTCH_stratGraphOrder(SCOTCH_Strat *, char *)
    int SCOTCH_stratGraphOrderBuild(SCOTCH_Strat *, SCOTCH_Num, double)
    int SCOTCH_stratMeshOrder(SCOTCH_Strat *, char *)
    int SCOTCH_stratMeshOrderBuild(SCOTCH_Strat *, SCOTCH_Num, double)

    void SCOTCH_memoryTrace()
    void SCOTCH_memoryUntrace()
    void SCOTCH_memoryTraceReset()
    unsigned long SCOTCH_memoryTraceGet()
