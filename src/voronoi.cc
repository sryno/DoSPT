#include <omp.h>
#include <iostream>
#include "voro++/install/include/voro++/voro++.hh"
using namespace voro;

extern "C" {

  void voronoi_volumes_(int *natoms_orig, int *nframes_orig, int *di_volumes_orig, double L[], float pos[], double volumes[]) {

    // voronoi.cc has been modified from the original code to work on 
    // a vectorized version of the position and volume input
    // It has been modifed to parallelize the put and compute 
    // commands via OpenMP
    int j2 = 1;
    int i, j, k;
    int natoms = *natoms_orig;
    int nframes = *nframes_orig;
    int di_volumes = *di_volumes_orig;
    int j2ndx[nframes] = {};
    int n_volumes;
    n_volumes = 1 + (nframes-1)/di_volumes;

    for (i=0; i < nframes; i++) {
      if (i == 0) {
        j2ndx[i] = j2;
        j2++;
      }
      else if (i % di_volumes == 0) {
        j2ndx[i] = j2;
        j2++;
      }
    }
    //This block should be parallelized in MPI
    {
      for (i=0; i < nframes; i++) {
        double volumes_temp[natoms];
        if (j2ndx[i] != 0) {
          double x[natoms];
          double y[natoms];
          double z[natoms];
          for (j=0; j<natoms; j++) {
            //pos[nframes*ncoords*atom# + coord#*nframes + frame#]
            x[j] = pos[nframes*3*j + 0*nframes + i];
            y[j] = pos[nframes*3*j + 1*nframes + i];
            z[j] = pos[nframes*3*j + 2*nframes + i];
          }
          container con(0.,L[0],0.,L[1],0.,L[2],1,1,1,true,true,true,8);
          #pragma omp for ordered schedule(dynamic)
          for (j=0; j<natoms; j++) {
            con.put(j, x[j], y[j], z[j]);
          }
          voronoicell v;
          #pragma omp for ordered schedule(dynamic)
          for (j=0; j<natoms; j++) {
            con.compute_cell(v,0,j);
            volumes_temp[j] = v.volume();
          }
          //This for loop move all data from volumes_temp[j] to volumes that the Fortran portion uses
          //This is the data collection from the child processes
          for (j=0; j<natoms; j++) {
            //Counting of n_vol# should start at 0, but the index starts at 1
            volumes[j*n_volumes+(j2ndx[i]-1)] = volumes_temp[j];
          }
        }
      }
    }
  }
}

//Ignore these. They are for reference.
/*x[row][col] = [row*(#cols) + col]*/
/*x[row][col][depth] = [depth*(#rows)*(#cols) + row*(#cols) + col]*/