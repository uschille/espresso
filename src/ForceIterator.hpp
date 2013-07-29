#ifndef FORCE_ITERATOR_HPP
#define FORCE_ITERATOR_HPP

#include "EspressoSystemInterface.hpp"
#include "OneParticleForce.hpp"
#include "statistics.hpp"
#include <vector>

class ForceIterator {
public:
  void addMethod(OneParticleForce *);
  void init();
  void run();
  void addForces();
  bool isReady();
  void runEnergies();
  void addEnergies(Observable_stat *);
  bool isReadyEnergies();
protected:
  EspressoSystemInterface System;
  std::vector<OneParticleForce *> methods;
  std::vector<SystemInterface::Vector3> F;
};

#endif
