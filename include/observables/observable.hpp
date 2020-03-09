#ifndef INCLUDE_OBSERVABLES_OBSERVABLE_HPP
#define INCLUDE_OBSERVABLES_OBSERVABLE_HPP

#include "properties.hpp"

namespace Observables {
namespace Observable {
/**
 * @brief Meta-Observable that returns the product of two
 *        other observables.
 *
 * The operand observables are stored by privately deriving
 * from them to get the empty case optimization if they do
 * not have state.
 *
 * @tparam Left left operand of the product.
 * @tparam Right right operand of the product.
 */
template <class Left, class Right> struct Product : Left, Right {
  Product(Left left = {}, Right right = {})
      : Left(std::move(left)), Right(std::move(right)) {}

  template <class Particle, class Traits = Properties::traits<Particle>>
  decltype(auto) operator()(Particle const &p,
                            Traits const &traits = {}) const {
    return Left::template operator()<Particle, Traits>(p, traits) *
           Right::template operator()<Particle, Traits>(p, traits);
  }
};

using Momentum = Product<Properties::Mass, Properties::Velocity>;
template <class Observable>
using Flux = Product<Observable, Properties::Velocity>;
using ElectricCurrent = Flux<Properties::Charge>;
} // namespace Observable
} // namespace Observables

#endif // INCLUDE_OBSERVABLES_OBSERVABLE_HPP
