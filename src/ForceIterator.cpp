#include "ForceIterator.hpp"
#include "cells.hpp"
#include "cuda_common.hpp"

#include <stdio.h>

void ForceIterator::addMethod(OneParticleForce *m) {
  methods.push_back(m);
}

void ForceIterator::init() {
  // turn on communication
  gpu_init_particle_comm();
  cuda_bcast_global_part_params();

  System.init();
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
    (*it)->init(System);
}

void ForceIterator::run() {
  System.update();
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
    (*it)->run(System);
}

bool ForceIterator::isReady() {
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
    if (!(*it)->isReady())
      return false;
  return true;
}

void ForceIterator::addForces() {
  Cell *cell;
  Particle *p;
  int i,c,np;

  if (methods.size() == 0)
    return;

  puts("Adding forces.");

  F.reserve(System.npart());

  for(unsigned int i = 0; i < System.npart(); i++)
    F[i] = SystemInterface::Vector3(0,0,0);



  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it) {
    int id = 0;
    for(SystemInterface::const_vec_iterator &jt = (*it)->fBegin(); jt != (*it)->fEnd(); ++jt) {
      F[id++] += (*jt);
    }
  }

  // send forces to other nodes
  CUDA_global_part_vars* global_part_vars_host = gpu_get_global_particle_vars_pointer_host();
  if ( global_part_vars_host->communication_enabled == 1 && global_part_vars_host->number_of_particles )
  {
    np = global_part_vars_host->number_of_particles;
    CUDA_particle_force *host_forces = (CUDA_particle_force *) malloc(np*sizeof(CUDA_particle_force));
    for (int i = 0; i < np; i++)
    {
      for (int d = 0; d < 3; d++)
        host_forces[i].f[d] = F[i][d];
    }
    cuda_mpi_send_forces(host_forces);
    free(host_forces);
  }
  else // only add forces to local node
  {
  for (c = 0; c < local_cells.n; c++) {
    cell = local_cells.cell[c];
    p  = cell->part;
    np = cell->n;
    for(i = 0; i < np; i++) {
      p[i].f.f[0] += F[i][0];
      p[i].f.f[1] += F[i][1];
      p[i].f.f[2] += F[i][2];
    }
  }
  }
}

