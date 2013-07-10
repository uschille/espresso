#ifndef ONEPARTICLESFORCE_H
#define ONEPARTICLESFORCE_H

#include "SystemInterface.hpp"

class OneParticleForce {
 public:
  virtual void init(SystemInterface &s) = 0;
  virtual void run(SystemInterface &s) = 0;
  virtual bool isReady() = 0;

  typedef SystemInterface::const_vec_iterator const_vec_iterator;

  virtual const_vec_iterator & fBegin() = 0;
  virtual const_vec_iterator & fEnd() = 0;
};

#endif

