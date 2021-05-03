#pragma once

namespace CppAD {
namespace cg {

/**
 * An atomic function for source code generation
 */
template <class Base>
class AbstractLoopAtomicFun : public CGAbstractAtomicFun<Base> {
 public:
  using Super = CGAbstractAtomicFun<Base>;
  using CGB = CppAD::cg::CG<Base>;
  using Arg = Argument<Base>;

 protected:
  /**
   * A unique identifier for this atomic function type
   */
  using Super::id_;

  const size_t num_iterations_;

  const size_t output_dim_;
  const size_t const_input_dim_;
  const size_t loop_dependent_dim_;

 protected:
  /**
   * Creates a new atomic function that is responsible for defining the
   * dependencies to calls of a user atomic function.
   *
   * @param name The atomic function name.
   * @param standAlone Whether or not forward and reverse function calls
   *                   do not require the Taylor coefficients for the
   *                   dependent variables (ty) and any previous
   *                   evaluation of other forward/reverse modes.
   */
  explicit AbstractLoopAtomicFun(const std::string& name, size_t num_iterations,
                                 size_t output_dim, size_t const_input_dim,
                                 size_t loop_dependent_dim)
      : Super(name, false),
        num_iterations_(num_iterations),
        output_dim_(output_dim),
        const_input_dim_(const_input_dim),
        loop_dependent_dim_(loop_dependent_dim) {}

 public:
  virtual ~AbstractLoopAtomicFun() = default;

  bool forward(size_t q, size_t p, const CppAD::vector<bool>& vx,
               CppAD::vector<bool>& vy, const CppAD::vector<CGB>& tx,
               CppAD::vector<CGB>& ty) override {
    using CppAD::vector;

    CppAD::vector<CGB> x;

    bool valuesDefined = BaseAbstractAtomicFun<Base>::isValuesDefined(tx);
    if (vx.size() > 0) {
      size_t n = vx.size();
      x.resize(n);
      for (size_t j = 0; j < n; j++) {
        x[j] = tx[j * (p + 1)];
      }

      zeroOrderDependency(vx, vy, x);
    }

    bool allParameters = BaseAbstractAtomicFun<Base>::isParameters(tx);
    if (allParameters) {
      vector<Base> tyb;
      if (!evalForwardValues(q, p, tx, tyb, ty.size())) return false;

      CPPADCG_ASSERT_UNKNOWN(tyb.size() == ty.size())
      for (size_t i = 0; i < ty.size(); i++) {
        ty[i] = tyb[i];
      }
      return true;
    }

    size_t m = ty.size() / (p + 1);

    vector<bool> vyLocal;
    if (p == 0) {
      vyLocal = vy;
    } else if (p >= 1) {
      /**
       * Use the jacobian sparsity to determine which elements
       * will always be zero
       */

      size_t n = tx.size() / (p + 1);

      vector<std::set<size_t>> r(n);
      for (size_t j = 0; j < n; j++) {
        if (!tx[j * (p + 1) + 1].isIdenticalZero()) r[j].insert(0);
      }
      vector<std::set<size_t>> s(m);

      if (x.size() == 0) {
        x.resize(n);
        for (size_t j = 0; j < n; j++) {
          x[j] = tx[j * (p + 1)];
        }
      }

      bool good = this->for_sparse_jac(1, r, s, x);
      if (!good) return false;

      vyLocal.resize(ty.size());
      for (size_t i = 0; i < vyLocal.size(); i++) {
        vyLocal[i] = true;
      }

      for (size_t i = 0; i < m; i++) {
        vyLocal[i * (p + 1) + 1] = !s[i].empty();
      }

      if (p == 1) {
        bool allZero = true;
        for (size_t i = 0; i < vyLocal.size(); i++) {
          if (vyLocal[i]) {
            allZero = false;
            break;
          }
        }

        if (allZero) {
          for (size_t i = 0; i < ty.size(); i++) {
            ty[i] = Base(0.0);
          }
          return true;
        }
      }
    }

    vector<Base> tyb;
    if (valuesDefined) {
      if (!evalForwardValues(q, p, tx, tyb, ty.size())) return false;
    }

    CodeHandler<Base>* handler = findHandler(tx);
    CPPADCG_ASSERT_UNKNOWN(handler != nullptr)

    /**
     * make the loop start
     */
    OperationNode<Base>* iterationIndexDcl =
        handler->makeIndexDclrNode(LoopModel<Base>::ITERATION_INDEX_NAME);
    LoopStartOperationNode<Base>* loopStart =
        handler->makeLoopStartNode(*iterationIndexDcl, num_iterations_);
    OperationNode<Base>* loopIndexNode = handler->makeIndexNode(*loopStart);

    std::vector<Argument<Base>>& startArgs = loopStart->getArguments();

    IndexOperationNode<Base>* index_node =
        handler->makeIndexNode(*iterationIndexDcl);

    size_t p1 = p + 1;

    std::vector<Arg> stateArrayArgs(tx.size() - loop_dependent_dim_);
    for (size_t i = 0; i < tx.size() - loop_dependent_dim_; i++) {
      stateArrayArgs[i] = asArgument(tx[i * (p + 1)]);
    }
    OperationNode<Base>* stateArray =
        handler->makeNode(CGOpCode::ArrayCreation, {}, stateArrayArgs);
    startArgs.push_back(*stateArray);

    std::vector<OperationNode<Base>*> txArray(p1), tyArray(p1);
    for (size_t k = 0; k < p1; k++) {
      if (k == 0) {
        size_t n = tx.size() / (p + 1);
        std::vector<Arg> arrayArgs(n);
        for (size_t i = 0; i < n - loop_dependent_dim_; i++) {
          OperationNode<Base>* state_element = handler->makeNode(
              CGOpCode::ArrayElement, {i}, {*stateArray, *loopStart});
          arrayArgs[i] = *state_element;
        }
        for (size_t i = 0; i < loop_dependent_dim_; ++i) {
          auto* index_pattern =
              new CppAD::cg::LinearIndexPattern(0, loop_dependent_dim_, 1, i);
          handler->addLoopIndependentIndexPattern(*index_pattern, i);
          OperationNode<Base>* xref =
              handler->makeNode(CGOpCode::LoopIndexedIndep, {0, i},
                                {*loopIndexNode, *index_node});
          arrayArgs[n - loop_dependent_dim_ + i] = *xref;
        }

        txArray[k] = handler->makeNode(CGOpCode::ArrayCreation, {}, arrayArgs);
      } else
        txArray[k] =
            BaseAbstractAtomicFun<Base>::makeSparseArray(*handler, tx, p, k);
      tyArray[k] = BaseAbstractAtomicFun<Base>::makeZeroArray(*handler, m);
    }

    // create atomic function call for the loop body
    std::vector<Argument<Base>> args(2 * p1);
    for (size_t k = 0; k < p1; k++) {
      args[0 * p1 + k] = *txArray[k];
      args[1 * p1 + k] = *tyArray[k];
    }
    OperationNode<Base>* atomicOp =
        handler->makeNode(CGOpCode::AtomicForward, {id_, q, p}, args);
    handler->registerAtomicFunction(*this);

    IndexOperationNode<Base>* iterationIndexOp =
        handler->makeIndexNode(*loopStart);
    std::set<IndexOperationNode<Base>*> indexesOps;
    indexesOps.insert(iterationIndexOp);

    /**
     * make the loop end
     */
    size_t assignOrAdd = 0;

    // operations in the loop body
    std::vector<Argument<Base>> endArgs;

    endArgs.push_back(*atomicOp);

    // assign loop body output to the input state at the next step
    for (size_t k = 0; k < p1; k++) {
      if (k == 0) {
        CppAD::vector<CGB> aty_cg(ty.size());
        for (size_t i = 0; i < ty.size(); ++i) {
          aty_cg[i] = handler->createCG(*handler->makeNode(
              CGOpCode::ArrayElement, {i}, {*tyArray[k], *atomicOp}));
        }
        OperationNode<Base>* assign_x =
            BaseAbstractAtomicFun<Base>::makeArray(*handler, aty_cg, p, k);
        endArgs.push_back(*assign_x);
      }
      //   }
      //   // else
      //   //   txArray[k] =
      //   //       BaseAbstractAtomicFun<Base>::makeSparseArray(*handler, tx,
      //   p, k);
      //   // tyArray[k] = BaseAbstractAtomicFun<Base>::makeZeroArray(*handler,
      //   m);
    }

    LoopEndOperationNode<Base>* loopEnd =
        handler->makeLoopEndNode(*loopStart, endArgs);

    handler->getLoopData().indexes.insert(iterationIndexDcl);

    // assign the output variables as the return state from the for-loop
    for (size_t i = 0; i < ty.size(); i++) {
      ty[i] = handler->createCG(*handler->makeNode(CGOpCode::ArrayElement, {i},
                                                   {*tyArray[0], *loopEnd}));
    }

    return true;
  }

  bool reverse(size_t p, const CppAD::vector<CGB>& tx,
               const CppAD::vector<CGB>& ty, CppAD::vector<CGB>& px,
               const CppAD::vector<CGB>& py) override {
    using CppAD::vector;

    bool allParameters = BaseAbstractAtomicFun<Base>::isParameters(tx);
    if (allParameters) {
      allParameters = BaseAbstractAtomicFun<Base>::isParameters(ty);
      if (allParameters) {
        allParameters = BaseAbstractAtomicFun<Base>::isParameters(py);
      }
    }

    if (allParameters) {
      vector<Base> pxb;

      if (!evalReverseValues(p, tx, ty, pxb, py)) return false;

      CPPADCG_ASSERT_UNKNOWN(pxb.size() == px.size())

      for (size_t i = 0; i < px.size(); i++) {
        px[i] = pxb[i];
      }
      return true;
    }

    /**
     * Use the Jacobian sparsity to determine which elements
     * will always be zero
     */
    vector<bool> vxLocal(px.size());
    for (size_t j = 0; j < vxLocal.size(); j++) {
      vxLocal[j] = true;
    }

    size_t p1 = p + 1;
    // k == 0
    size_t m = ty.size() / p1;
    size_t n = tx.size() / p1;

    vector<std::set<size_t>> rt(m);
    for (size_t i = 0; i < m; i++) {
      if (!py[i * p1].isIdenticalZero()) {
        rt[i].insert(0);
      }
    }

    CppAD::vector<CGB> x(n);
    for (size_t j = 0; j < n; j++) {
      x[j] = tx[j * p1];
    }

    vector<std::set<size_t>> st(n);
    bool good = this->rev_sparse_jac(1, rt, st, x);
    if (!good) {
      return false;
    }

    for (size_t j = 0; j < n; j++) {
      vxLocal[j * p1 + p] = !st[j].empty();
    }

    if (p >= 1) {
      /**
       * Use the Hessian sparsity to determine which elements
       * will always be zero
       */
      vector<bool> vx(n);
      vector<bool> s(m);
      vector<bool> t(n);
      vector<std::set<size_t>> r(n);
      vector<std::set<size_t>> u(m);
      vector<std::set<size_t>> v(n);

      for (size_t j = 0; j < n; j++) {
        vx[j] = !tx[j * p1].isParameter();
        if (!tx[j * p1 + 1].isIdenticalZero()) {
          r[j].insert(0);
        }
      }
      for (size_t i = 0; i < m; i++) {
        s[i] = !py[i * p1 + 1].isIdenticalZero();
      }

      this->rev_sparse_hes(vx, s, t, 1, r, u, v, x);

      for (size_t j = 0; j < n; j++) {
        vxLocal[j * p1 + p - 1] = !v[j].empty();
      }
    }

    bool allZero = true;
    for (size_t j = 0; j < vxLocal.size(); j++) {
      if (vxLocal[j]) {
        allZero = false;
        break;
      }
    }

    if (allZero) {
      for (size_t j = 0; j < px.size(); j++) {
        px[j] = Base(0.0);
      }
      return true;
    }

    bool valuesDefined = BaseAbstractAtomicFun<Base>::isValuesDefined(tx);
    if (valuesDefined) {
      valuesDefined = BaseAbstractAtomicFun<Base>::isValuesDefined(ty);
      if (valuesDefined) {
        valuesDefined = BaseAbstractAtomicFun<Base>::isValuesDefined(py);
      }
    }

    vector<Base> pxb;
    if (valuesDefined) {
      if (!evalReverseValues(p, tx, ty, pxb, py)) return false;
    }

    CodeHandler<Base>* handler = findHandler(tx);
    if (handler == nullptr) {
      handler = findHandler(ty);
      if (handler == nullptr) {
        handler = findHandler(py);
      }
    }
    CPPADCG_ASSERT_UNKNOWN(handler != nullptr)

    std::vector<OperationNode<Base>*> txArray(p1), tyArray(p1), pxArray(p1),
        pyArray(p1);
    for (size_t k = 0; k <= p; k++) {
      if (k == 0)
        txArray[k] = BaseAbstractAtomicFun<Base>::makeArray(*handler, tx, p, k);
      else
        txArray[k] =
            BaseAbstractAtomicFun<Base>::makeSparseArray(*handler, tx, p, k);

      if (standAlone_) {
        tyArray[k] =
            BaseAbstractAtomicFun<Base>::makeEmptySparseArray(*handler, m);
      } else {
        tyArray[k] =
            BaseAbstractAtomicFun<Base>::makeSparseArray(*handler, ty, p, k);
      }

      if (k == 0)
        pxArray[k] = BaseAbstractAtomicFun<Base>::makeZeroArray(*handler, n);
      else
        pxArray[k] =
            BaseAbstractAtomicFun<Base>::makeEmptySparseArray(*handler, n);

      if (k == 0)
        pyArray[k] =
            BaseAbstractAtomicFun<Base>::makeSparseArray(*handler, py, p, k);
      else
        pyArray[k] = BaseAbstractAtomicFun<Base>::makeArray(*handler, py, p, k);
    }

    std::vector<Argument<Base>> args(4 * p1);
    for (size_t k = 0; k <= p; k++) {
      args[0 * p1 + k] = *txArray[k];
      args[1 * p1 + k] = *tyArray[k];
      args[2 * p1 + k] = *pxArray[k];
      args[3 * p1 + k] = *pyArray[k];
    }

    OperationNode<Base>* atomicOp =
        handler->makeNode(CGOpCode::AtomicReverse, {id_, p}, args);
    handler->registerAtomicFunction(*this);

    for (size_t k = 0; k < p1; k++) {
      for (size_t j = 0; j < n; j++) {
        size_t pos = j * p1 + k;
        if (vxLocal[pos]) {
          px[pos] = CGB(*handler->makeNode(CGOpCode::ArrayElement, {j},
                                           {*pxArray[k], *atomicOp}));
          if (valuesDefined) {
            px[pos].setValue(pxb[pos]);
          }
        } else {
          // CPPADCG_ASSERT_KNOWN(pxb.size() == 0 || IdenticalZero(pxb[j]),
          // "Invalid value"); pxb[j] might be non-zero but it is not required
          // (it might have been used to determine other pxbs)
          px[pos] = Base(0);  // not a variable (zero)
        }
      }
    }

    return true;
  }

  inline virtual CppAD::vector<std::set<size_t>> jacobianForwardSparsitySet(
      size_t m, const CppAD::vector<CGB>& x) {
    size_t n = x.size();

    CppAD::vector<std::set<size_t>> r(n);  // identity matrix
    for (size_t i = 0; i < n; i++) r[i].insert(i);

    CppAD::vector<std::set<size_t>> s(m);
    bool good = this->for_sparse_jac(n, r, s, x);
    if (!good)
      throw CGException(
          "Failed to compute jacobian sparsity pattern for atomic function '",
          this->atomic_name(), "'");

    return s;
  }

  inline virtual CppAD::vector<std::set<size_t>> jacobianReverseSparsitySet(
      size_t m, const CppAD::vector<CGB>& x) {
    size_t n = x.size();

    CppAD::vector<std::set<size_t>> rt(m);  // identity matrix
    for (size_t i = 0; i < m; i++) rt[i].insert(i);

    CppAD::vector<std::set<size_t>> st(n);
    bool good = this->rev_sparse_jac(m, rt, st, x);
    if (!good)
      throw CGException(
          "Failed to compute jacobian sparsity pattern for atomic function '",
          this->atomic_name(), "'");

    CppAD::vector<std::set<size_t>> s = transposePattern(st, n, m);

    return s;
  }

  inline virtual CppAD::vector<std::set<size_t>> hessianSparsitySet(
      size_t m, const CppAD::vector<CGB>& x) {
    CppAD::vector<bool> s(m);
    for (size_t i = 0; i < m; ++i) s[i] = true;

    return hessianSparsitySet(s, x);
  }

  inline virtual CppAD::vector<std::set<size_t>> hessianSparsitySet(
      const CppAD::vector<bool>& s, const CppAD::vector<CGB>& x) {
    size_t n = x.size();
    size_t m = s.size();

    /**
     * Determine the sparsity pattern p for Hessian of w^T F
     */
    CppAD::vector<std::set<size_t>> r(n);  // identity matrix
    for (size_t j = 0; j < n; j++) r[j].insert(j);

    CppAD::vector<bool> vx(n);  // which x's are variables
    for (size_t i = 0; i < n; ++i) vx[i] = true;

    CppAD::vector<bool> t(n);
    for (size_t i = 0; i < n; ++i) t[i] = false;

    const CppAD::vector<std::set<size_t>> u(m);  // empty
    CppAD::vector<std::set<size_t>> v(n);

    bool good = this->rev_sparse_hes(vx, s, t, n, r, u, v, x);
    if (!good)
      throw CGException(
          "Failed to compute Hessian sparsity pattern for atomic function '",
          this->atomic_name(), "'");

    return v;
  }

  /**
   * Uses an internal counter to produce IDs for atomic functions.
   */
  static size_t createNewAtomicFunctionID() {
    CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
    static size_t count = 0;
    count++;
    return count;
  }

 protected:
  virtual void zeroOrderDependency(const CppAD::vector<bool>& vx,
                                   CppAD::vector<bool>& vy,
                                   const CppAD::vector<CGB>& x) = 0;

  /**
   * Used to evaluate function values and forward mode function values and
   * derivatives.
   *
   * @param q Lowest order for this forward mode calculation.
   * @param p Highest order for this forward mode calculation.
   * @param vx If size not zero, which components of \c x are variables
   * @param vy If size not zero, which components of \c y are variables
   * @param tx Taylor coefficients corresponding to \c x for this
   *           calculation
   * @param ty Taylor coefficient corresponding to \c y for this
   *           calculation
   * @return true on success, false otherwise
   */
  virtual bool atomicForward(size_t q, size_t p, const CppAD::vector<Base>& tx,
                             CppAD::vector<Base>& ty) = 0;
  /**
   * Used to evaluate reverse mode function derivatives.
   *
   * @param p Highest order for this forward mode calculation.
   * @param tx Taylor coefficients corresponding to \c x for this
   *           calculation
   * @param ty Taylor coefficient corresponding to \c y for this
   *           calculation
   * @param px Partials w.r.t. the \c x Taylor coefficients.
   * @param py Partials w.r.t. the \c y Taylor coefficients
   * @return true on success, false otherwise
   */
  virtual bool atomicReverse(size_t p, const CppAD::vector<Base>& tx,
                             const CppAD::vector<Base>& ty,
                             CppAD::vector<Base>& px,
                             const CppAD::vector<Base>& py) = 0;

 private:
  inline bool evalForwardValues(size_t q, size_t p,
                                const CppAD::vector<CGB>& tx,
                                CppAD::vector<Base>& tyb, size_t ty_size) {
    CppAD::vector<Base> txb(tx.size());
    tyb.resize(ty_size);

    for (size_t i = 0; i < txb.size(); i++) {
      txb[i] = tx[i].getValue();
    }

    return atomicForward(q, p, txb, tyb);
  }

  inline bool evalReverseValues(size_t p, const CppAD::vector<CGB>& tx,
                                const CppAD::vector<CGB>& ty,
                                CppAD::vector<Base>& pxb,
                                const CppAD::vector<CGB>& py) {
    using CppAD::vector;

    vector<Base> txb(tx.size());
    vector<Base> tyb(ty.size());
    pxb.resize(tx.size());
    vector<Base> pyb(py.size());

    for (size_t i = 0; i < txb.size(); i++) {
      txb[i] = tx[i].getValue();
    }
    for (size_t i = 0; i < tyb.size(); i++) {
      tyb[i] = ty[i].getValue();
    }
    for (size_t i = 0; i < pyb.size(); i++) {
      pyb[i] = py[i].getValue();
    }

    return atomicReverse(p, txb, tyb, pxb, pyb);
  }
};

}  // namespace cg
}  // namespace CppAD
