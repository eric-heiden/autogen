#pragma once

#include "cppad/cg/extra/sparsity.hpp"
#include "loop_atomic_fun.hpp"

namespace CppAD {
namespace cg {

/**
 * An atomic function wrapper for atomic functions using the ::CppAD::cg::CG
 * type.
 * This class can be useful when a CppAD::ADFun<CppAD::cg::CG> is going to
 * be used to create a compiled model library but has not been compiled yet.
 */
template <class Base>
class LoopFunBridge : public AbstractLoopAtomicFun<Base> {
 public:
  using CGB = CppAD::cg::CG<Base>;
  using ADCGD = CppAD::AD<CGB>;

 protected:
  ADFun<CGB>& fun_;
  bool cacheSparsities_;
  CustomPosition custom_jac_;
  CustomPosition custom_hess_;
  std::map<size_t, CppAD::vector<std::set<size_t> > > hess_;
  const size_t num_iterations;

 public:
  /**
   * Creates a new atomic function wrapper.
   *
   * @param name The atomic function name
   * @param fun The atomic function to be wrapped
   * @param cacheSparsities Whether or not to cache information related
   *                        with sparsity evaluation.
   */
  LoopFunBridge(const std::string& name, CppAD::ADFun<CGB>& fun,
                size_t num_iterations, size_t const_input_dim,
                size_t loop_dependent_dim, bool cacheSparsities = true)
      : AbstractLoopAtomicFun<Base>(name, num_iterations, fun.Range(),
                                    const_input_dim, loop_dependent_dim),
        fun_(fun),
        cacheSparsities_(cacheSparsities),
        num_iterations(num_iterations) {
    this->option(CppAD::atomic_base<CGB>::set_sparsity_enum);
  }

  LoopFunBridge(const LoopFunBridge& orig) = delete;
  LoopFunBridge& operator=(const LoopFunBridge& rhs) = delete;

  virtual ~LoopFunBridge() = default;

  template <class ADVector>
  void operator()(const ADVector& ax, ADVector& ay, size_t id = 0) {
    this->AbstractLoopAtomicFun<Base>::operator()(ax, ay, id);
  }

  template <class VectorSize>
  inline void setCustomSparseJacobianElements(const VectorSize& row,
                                              const VectorSize& col) {
    custom_jac_ = CustomPosition(fun_.Range(), fun_.Domain(), row, col);
  }

  template <class VectorSet>
  inline void setCustomSparseJacobianElements(const VectorSet& elements) {
    custom_jac_ = CustomPosition(fun_.Range(), fun_.Domain(), elements);
  }

  template <class VectorSize>
  inline void setCustomSparseHessianElements(const VectorSize& row,
                                             const VectorSize& col) {
    size_t n = fun_.Domain();
    custom_hess_ = CustomPosition(n, n, row, col);
  }

  template <class VectorSet>
  inline void setCustomSparseHessianElements(const VectorSet& elements) {
    size_t n = fun_.Domain();
    custom_hess_ = CustomPosition(n, n, elements);
  }

  bool for_sparse_jac(size_t q, const CppAD::vector<std::set<size_t> >& r,
                      CppAD::vector<std::set<size_t> >& s,
                      const CppAD::vector<CGB>& x) override {
    return for_sparse_jac(q, r, s);
  }

  bool for_sparse_jac(size_t q, const CppAD::vector<std::set<size_t> >& r,
                      CppAD::vector<std::set<size_t> >& s) override {
    using CppAD::vector;

    if (cacheSparsities_ || custom_jac_.isFilterDefined()) {
      size_t n = fun_.Domain();
      size_t m = fun_.Range();
      if (!custom_jac_.isFullDefined()) {
        custom_jac_.setFullElements(CppAD::cg::jacobianForwardSparsitySet<
                                    std::vector<std::set<size_t> >, CGB>(fun_));
        fun_.size_forward_set(0);
      }

      for (size_t i = 0; i < s.size(); i++) {
        s[i].clear();
      }
      CppAD::cg::multMatrixMatrixSparsity(custom_jac_.getFullElements(), r, s,
                                          m, n, q);
    } else {
      s = fun_.ForSparseJac(q, r);
      fun_.size_forward_set(0);
    }

    return true;
  }

  bool rev_sparse_jac(size_t q, const CppAD::vector<std::set<size_t> >& rt,
                      CppAD::vector<std::set<size_t> >& st,
                      const CppAD::vector<CGB>& x) override {
    return rev_sparse_jac(q, rt, st);
  }

  bool rev_sparse_jac(size_t q, const CppAD::vector<std::set<size_t> >& rt,
                      CppAD::vector<std::set<size_t> >& st) override {
    using CppAD::vector;

    if (cacheSparsities_ || custom_jac_.isFilterDefined()) {
      size_t n = fun_.Domain();
      size_t m = fun_.Range();
      if (!custom_jac_.isFullDefined()) {
        custom_jac_.setFullElements(CppAD::cg::jacobianReverseSparsitySet<
                                    std::vector<std::set<size_t> >, CGB>(fun_));
      }

      for (size_t i = 0; i < st.size(); i++) {
        st[i].clear();
      }
      CppAD::cg::multMatrixMatrixSparsityTrans(
          rt, custom_jac_.getFullElements(), st, m, n, q);
    } else {
      st = fun_.RevSparseJac(q, rt, true);
    }

    return true;
  }

  bool rev_sparse_hes(const CppAD::vector<bool>& vx,
                      const CppAD::vector<bool>& s, CppAD::vector<bool>& t,
                      size_t q, const CppAD::vector<std::set<size_t> >& r,
                      const CppAD::vector<std::set<size_t> >& u,
                      CppAD::vector<std::set<size_t> >& v,
                      const CppAD::vector<CGB>& x) override {
    return rev_sparse_hes(vx, s, t, q, r, u, v);
  }

  bool rev_sparse_hes(const CppAD::vector<bool>& vx,
                      const CppAD::vector<bool>& s, CppAD::vector<bool>& t,
                      size_t q, const CppAD::vector<std::set<size_t> >& r,
                      const CppAD::vector<std::set<size_t> >& u,
                      CppAD::vector<std::set<size_t> >& v) override {
    using CppAD::vector;

    if (cacheSparsities_ || custom_jac_.isFilterDefined() ||
        custom_hess_.isFilterDefined()) {
      size_t n = fun_.Domain();
      size_t m = fun_.Range();

      for (size_t i = 0; i < n; i++) {
        v[i].clear();
      }

      if (!custom_jac_.isFullDefined()) {
        custom_jac_.setFullElements(
            CppAD::cg::jacobianSparsitySet<std::vector<std::set<size_t> >, CGB>(
                fun_));
      }
      const std::vector<std::set<size_t> >& jacSparsity =
          custom_jac_.getFullElements();

      /**
       *  V(x)  =  f'^T(x) U(x)  +  Sum(  s(x)i  f''(x)  R(x)   )
       */
      // f'^T(x) U(x)
      CppAD::cg::multMatrixTransMatrixSparsity(jacSparsity, u, v, m, n, q);

      // Sum(  s(x)i  f''(x)  R(x)   )
      bool allSelected = true;
      for (size_t i = 0; i < m; i++) {
        if (!s[i]) {
          allSelected = false;
          break;
        }
      }

      if (allSelected) {
        if (!custom_hess_.isFullDefined()) {
          custom_hess_.setFullElements(
              CppAD::cg::hessianSparsitySet<std::vector<std::set<size_t> >,
                                            CGB>(fun_));  // f''(x)
        }
        const std::vector<std::set<size_t> >& sF2 =
            custom_hess_.getFullElements();
        CppAD::cg::multMatrixTransMatrixSparsity(sF2, r, v, n, n,
                                                 q);  // f''^T * R
      } else {
        CppAD::vector<std::set<size_t> > sparsitySF2R(n);
        for (size_t i = 0; i < m; i++) {
          if (s[i]) {
            const auto itH = hess_.find(i);
            const CppAD::vector<std::set<size_t> >* spari;
            if (itH == hess_.end()) {
              hess_[i] = CppAD::cg::hessianSparsitySet<
                  CppAD::vector<std::set<size_t> >, CGB>(fun_, i);  // f''_i(x)
              spari = &hess_[i];
              custom_hess_.filter(hess_[i]);
            } else {
              spari = &itH->second;
            }
            CppAD::cg::addMatrixSparsity(*spari, sparsitySF2R);
          }
        }
        CppAD::cg::multMatrixTransMatrixSparsity(sparsitySF2R, r, v, n, n,
                                                 q);  // f''^T * R
      }

      /**
       * S(x) * f'(x)
       */
      for (size_t i = 0; i < m; i++) {
        if (s[i]) {
          for (size_t j : jacSparsity[i]) {
            t[j] = true;
          }
        }
      }
    } else {
      size_t m = fun_.Range();
      size_t n = fun_.Domain();

      t = fun_.RevSparseJac(1, s);
      vector<std::set<size_t> > a = fun_.RevSparseJac(q, u, true);

      // set version of s
      vector<std::set<size_t> > set_s(1);
      for (size_t i = 0; i < m; i++) {
        if (s[i]) set_s[0].insert(i);
      }

      fun_.ForSparseJac(q, r);
      v = fun_.RevSparseHes(q, set_s, true);

      for (size_t i = 0; i < n; i++) {
        for (size_t j : a[i]) {
          CPPAD_ASSERT_UNKNOWN(j < q)
          v[i].insert(j);
        }
      }

      fun_.size_forward_set(0);
    }

    return true;
  }

 protected:
  void zeroOrderDependency(const CppAD::vector<bool>& vx,
                           CppAD::vector<bool>& vy,
                           const CppAD::vector<CGB>& x) override {
    CppAD::cg::zeroOrderDependency(fun_, vx, vy);
  }

  bool atomicForward(size_t q, size_t p, const CppAD::vector<Base>& tx,
                     CppAD::vector<Base>& ty) override {
    using CppAD::vector;

    vector<CGB> txcg(tx.size());
    toCG(tx, txcg);

    vector<CGB> tycg = fun_.Forward(p, txcg);
    fromCG(tycg, ty);

    fun_.capacity_order(0);

    return true;
  }

  bool atomicReverse(size_t p, const CppAD::vector<Base>& tx,
                     const CppAD::vector<Base>& ty, CppAD::vector<Base>& px,
                     const CppAD::vector<Base>& py) override {
    using CppAD::vector;

    vector<CGB> txcg(tx.size());
    vector<CGB> pycg(py.size());

    toCG(tx, txcg);
    toCG(py, pycg);

    fun_.Forward(p, txcg);

    vector<CGB> pxcg = fun_.Reverse(p + 1, pycg);
    fromCG(pxcg, px);

    fun_.capacity_order(0);
    return true;
  }

 private:
  static void toCG(const CppAD::vector<Base>& from, CppAD::vector<CGB>& to) {
    CPPAD_ASSERT_UNKNOWN(from.size() == to.size())

    for (size_t i = 0; i < from.size(); i++) {
      to[i] = from[i];
    }
  }

  static void fromCG(const CppAD::vector<CGB>& from, CppAD::vector<Base>& to) {
    CPPAD_ASSERT_UNKNOWN(from.size() == to.size())

    for (size_t i = 0; i < from.size(); i++) {
      CPPADCG_ASSERT_KNOWN(from[i].isValueDefined(), "No value defined")
      to[i] = from[i].getValue();
    }
  }
};

}  // namespace cg
}  // namespace CppAD
