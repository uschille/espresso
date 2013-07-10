#ifndef VECTORFORCE_HPP
#define VECTORFORCE_HPP

#include "OneParticleForce.hpp"
#include "SystemInterface.hpp"
#include "Eigen/Core"

class VectorForce : public OneParticleForce {
public:
  typedef std::vector<SystemInterface::Vector3> Vector3Container;
  
  typedef SystemInterface::const_iterator_stl<Vector3Container> const_vec_iterator;
  
  VectorForce() {
  };
  const_vec_iterator & fBegin() {
    m_f_begin = SystemInterface::const_iterator_stl<Vector3Container>(F.begin());
    return m_f_begin;
  };
  const_vec_iterator & fEnd() {
    m_f_end = SystemInterface::const_iterator_stl<Vector3Container>(F.end());
    return m_f_end;
  };
protected:
  Vector3Container F;

  SystemInterface::const_iterator_stl<Vector3Container> m_f_begin, m_f_end;
};

#endif
