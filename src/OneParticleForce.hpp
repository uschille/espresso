#ifndef ONEPARTICLESFORCE_H
#define ONEPARTICLESFORCE_H

#include "SystemInterface.hpp"
#include "statistics.hpp"

class OneParticleForce {
 public:
  virtual void init(SystemInterface &s) = 0;
  virtual void run(SystemInterface &s) = 0;
  virtual bool isReady() = 0;
  virtual void runEnergies(SystemInterface &s) = 0;
  virtual bool isReadyEnergies() = 0;

  typedef SystemInterface::const_vec_iterator const_vec_iterator;

  virtual const_vec_iterator & fBegin() = 0;
  virtual const_vec_iterator & fEnd() = 0;

  Observable_stat energy;
};

#endif

