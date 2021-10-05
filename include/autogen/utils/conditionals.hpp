#pragma once

#undef min
#undef max

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

namespace autogen {
template <typename Scalar>
static CppAD::AD<Scalar> where_gt(const CppAD::AD<Scalar>& x,
                                  const CppAD::AD<Scalar>& y,
                                  const CppAD::AD<Scalar>& if_true,
                                  const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpGt(x, y, if_true, if_false);
}

template <typename Scalar>
static CppAD::AD<Scalar> where_ge(const CppAD::AD<Scalar>& x,
                                  const CppAD::AD<Scalar>& y,
                                  const CppAD::AD<Scalar>& if_true,
                                  const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpGe(x, y, if_true, if_false);
}

template <typename Scalar>
static CppAD::AD<Scalar> where_lt(const CppAD::AD<Scalar>& x,
                                  const CppAD::AD<Scalar>& y,
                                  const CppAD::AD<Scalar>& if_true,
                                  const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpLt(x, y, if_true, if_false);
}
template <typename Scalar>
static CppAD::AD<Scalar> where_le(const CppAD::AD<Scalar>& x,
                                  const CppAD::AD<Scalar>& y,
                                  const CppAD::AD<Scalar>& if_true,
                                  const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpLe(x, y, if_true, if_false);
}
template <typename Scalar>
static CppAD::AD<Scalar> where_eq(const CppAD::AD<Scalar>& x,
                                  const CppAD::AD<Scalar>& y,
                                  const CppAD::AD<Scalar>& if_true,
                                  const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpEq(x, y, if_true, if_false);
}
template <typename Scalar>
static CppAD::AD<Scalar> isnan(const CppAD::AD<Scalar>& x,
                               const CppAD::AD<Scalar>& if_true,
                               const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpEq(x, x, if_false, if_true);
}

template <typename Scalar>
static FORCE_INLINE Scalar where_gt(const Scalar& x, const Scalar& y,
                                    const Scalar& if_true,
                                    const Scalar& if_false) {
  return x > y ? if_true : if_false;
}

template <typename Scalar>
static FORCE_INLINE Scalar where_ge(const Scalar& x, const Scalar& y,
                                    const Scalar& if_true,
                                    const Scalar& if_false) {
  return x >= y ? if_true : if_false;
}

template <typename Scalar>
static FORCE_INLINE Scalar where_lt(const Scalar& x, const Scalar& y,
                                    const Scalar& if_true,
                                    const Scalar& if_false) {
  return x < y ? if_true : if_false;
}

template <typename Scalar>
static FORCE_INLINE Scalar where_le(const Scalar& x, const Scalar& y,
                                    const Scalar& if_true,
                                    const Scalar& if_false) {
  return x <= y ? if_true : if_false;
}

template <typename Scalar>
static FORCE_INLINE Scalar where_eq(const Scalar& x, const Scalar& y,
                                    const Scalar& if_true,
                                    const Scalar& if_false) {
  return x == y ? if_true : if_false;
}

template <typename Scalar>
static FORCE_INLINE Scalar isnan(const Scalar& x, const Scalar& if_true,
                                 const Scalar& if_false) {
  return x == x ? if_false : if_true;
}

template <typename Scalar>
static FORCE_INLINE Scalar min(const Scalar& x, const Scalar& y) {
  return where_lt(x, y, x, y);
}

template <typename Scalar>
static FORCE_INLINE Scalar max(const Scalar& x, const Scalar& y) {
  return where_gt(x, y, x, y);
}
}  // namespace autogen