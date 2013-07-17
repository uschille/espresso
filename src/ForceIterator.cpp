#include "ForceIterator.hpp"
#include "cells.hpp"

#include <stdio.h>

void ForceIterator::addMethod(OneParticleForce *m) {
  methods.push_back(m);
}

void ForceIterator::init() {
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

