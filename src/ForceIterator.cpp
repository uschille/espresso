#include "ForceIterator.hpp"
#include "cells.hpp"

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

void ForceIterator::addForces() {
  Cell *cell;
  Particle *p;
  int i,c,np;

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

