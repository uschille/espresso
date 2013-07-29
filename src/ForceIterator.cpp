#include "ForceIterator.hpp"
#include "cells.hpp"
#include "cuda_common.hpp"
#include "energy.hpp"

#include <stdio.h>

void ForceIterator::addMethod(OneParticleForce *m) {
  m->energy.data.n = 0;
  m->energy.data.e = NULL;
  m->energy.data.max = 0;
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

void ForceIterator::runEnergies() {
  System.update();
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
  {
    init_energies(&(*it)->energy);
    (*it)->runEnergies(System);
  }
}

bool ForceIterator::isReady() {
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
    if (!(*it)->isReady())
      return false;
  return true;
}

bool ForceIterator::isReadyEnergies() {
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
    if (!(*it)->isReadyEnergies())
      return false;
  return true;
}

void ForceIterator::addForces() {
  Cell *cell;
  Particle *p;
  int i,c,np;

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
  if ( global_part_vars_host->communication_enabled == 1)
  {
    np = global_part_vars_host->number_of_particles;
    CUDA_particle_force *host_forces = (CUDA_particle_force *) malloc(np*sizeof(CUDA_particle_force));
    if (this_node == 0) for (int i = 0; i < np; i++)
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

void ForceIterator::addEnergies(Observable_stat *energy) {
  for(std::vector<OneParticleForce *>::const_iterator it = methods.begin(); it != methods.end(); ++it)
  {
    for (int i = 0; i < 1; i++)
      energy->bonded[i] += (*it)->energy.bonded[i];
    for (int i = 0; i < energy->n_non_bonded; i++)
      energy->non_bonded[i] += (*it)->energy.non_bonded[i];
    for (int i = 0; i < energy->n_coulomb; i++)
      energy->coulomb[i] += (*it)->energy.coulomb[i];
    for (int i = 0; i < energy->n_dipolar; i++)
      energy->dipolar[i] += (*it)->energy.dipolar[i];
  }
}