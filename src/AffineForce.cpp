#include "AffineForce.hpp"
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
