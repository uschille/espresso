#include "AffineForce.hpp"
AffineForce::AffineForce() {
  m_A = Eigen::Matrix3d::Identity();
  m_w = Eigen::Vector3d(1.,2.,3.);
}

AffineForce::AffineForce(Eigen::Matrix3d A, Eigen::Vector3d w) {
  m_A = A;
  m_w = w;
}

void AffineForce::run(SystemInterface &s) {
  F.reserve(s.npart());

  for(SystemInterface::const_vec_iterator &it_r = s.rBegin(); it_r != s.rEnd(); ++it_r) {
    F.push_back( m_A*(*it_r) + m_w );
  }
}
