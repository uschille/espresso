#ifndef AFFINEFORCE_H
#define AFFINEFORCE_H

#include "VectorForce.hpp"

class AffineForce : public VectorForce {
public:
  AffineForce(Eigen::Matrix3d A, Eigen::Vector3d w);
  void init(SystemInterface &s) {
    F.reserve(s.npart());
  };
  void run(SystemInterface &s);
  bool isReady() { return true; }
protected:
  Eigen::Matrix3d m_A;
  Eigen::Vector3d m_w;
};

#endif
