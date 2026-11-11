/*
  Copyright 2016 IRIS AS
  Copyright 2019, 2020 Equinor ASA
  Copyright 2020 SINTEF

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_WELLOPERATORS_HEADER_INCLUDED
#define OPM_WELLOPERATORS_HEADER_INCLUDED

#include <dune/common/parallel/communication.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <opm/common/TimingMacros.hpp>

#include <opm/simulators/linalg/matrixblock.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/istl/paamg/smoother.hh>

#include <cstddef>
#include <type_traits>

namespace Opm {

//=====================================================================
// Implementation for ISTL-matrix based operators
// Note: the classes WellModelMatrixAdapter and
// WellModelGhostLastMatrixAdapter were moved from ISTLSolver.hpp
// and subsequently modified.
//=====================================================================

/// Linear operator wrapper for well model.
///
/// This class is intended to hide the actual type of the well model
/// (which depends on a TypeTag) by making a simple linear operator
/// wrapper. That way the WellModelMatrixAdapter template does not need
/// the concrete WellModel type, and we can avoid instantiating
/// WellModelMatrixAdapter separately for each TypeTag, as it will only
/// depend on the matrix and vector types involved, which typically are
/// just one for each block size with block sizes 1-4.
template <class X, class Y>
class LinearOperatorExtra : public Dune::LinearOperator<X, Y>
{
public:
    using field_type = typename X::field_type;
    using PressureMatrix = Dune::BCRSMatrix<MatrixBlock<field_type, 1, 1>>;
    virtual void addWellPressureEquations(PressureMatrix& jacobian,
                                          const X& weights,
                                          const bool use_well_weights,
                                          const bool contract_d_diagonal) const = 0;
    virtual void addWellPressureEquationsStruct(PressureMatrix& jacobian) const = 0;
    virtual int getNumberOfExtraEquations() const = 0;

    //! \brief Two-phase application: applyStart(x) may start communication (for distributed
    //! wells), and must be followed by applyFinish(x, y) (same result as apply(x, y)) or
    //! applyscaleaddFinish(alpha, x, y) (same result as applyscaleadd(alpha, x, y)). Work not
    //! involving this operator (e.g. the reservoir matrix product) may be done in between.
    //! The default does everything in the finishing call.
    virtual void applyStart(const X& /*x*/) const {}
    virtual void applyFinish(const X& x, Y& y) const { this->apply(x, y); }
    virtual void applyscaleaddFinish(field_type alpha, const X& x, Y& y) const
    {
        this->applyscaleadd(alpha, x, y);
    }

    //! \brief As applyscaleadd(), but distributed wells only use the perforations on this
    //! process, so that no communication is needed. Intended for use in preconditioners.
    //! The default is applyscaleadd().
    virtual void applyscaleaddLocal(field_type alpha, const X& x, Y& y) const
    {
        this->applyscaleadd(alpha, x, y);
    }
};

//! \brief Wells supporting the two-phase (and communication-free local) operator application.
template<class Well, class V>
concept TwoPhaseApplyWell = requires(const Well& well, const V& x, V& y)
{
    well.applyStart(x);
    well.applyFinish(y);
    well.applyLocal(x, y);
};

template <class WellModel, class X, class Y>
class WellModelAsLinearOperator : public LinearOperatorExtra<X, Y>
{
public:
    using Base = LinearOperatorExtra<X, Y>;
    using field_type = typename Base::field_type;
    using PressureMatrix = typename Base::PressureMatrix;
    explicit WellModelAsLinearOperator(const WellModel& wm)
        : wellMod_(wm)
    {
    }

    /*! \brief apply operator to x:  \f$ y = A(x) \f$
       The input vector is consistent and the output must also be
       consistent on the interior+border partition.
     */
    void apply(const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(apply);
        applyStart(x);
        applyFinish(x, y);
    }

    //! apply operator to x, scale and add:  \f$ y = y + \alpha A(x) \f$
    void applyscaleadd(field_type alpha, const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(applyscaleadd);
        applyStart(x);
        applyscaleaddFinish(alpha, x, y);
    }

    //! Computes the local B x of all wells and starts the reductions for distributed wells,
    //! so that they can proceed concurrently and overlap with other work.
    void applyStart(const X& x) const override
    {
        OPM_TIMEBLOCK(applyStart);
        for (const auto& well : this->wellMod_) {
            if constexpr (TwoPhaseApplyWell<std::remove_cvref_t<decltype(*well)>, X>) {
                const auto& cells = well->cells();
                x_local_.resize(cells.size());
                for (std::size_t i = 0; i < cells.size(); ++i) {
                    x_local_[i] = x[cells[i]];
                }
                well->applyStart(x_local_);
            }
        }
    }

    void applyFinish(const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(applyFinish);
        for (const auto& well : this->wellMod_) {
            if constexpr (TwoPhaseApplyWell<std::remove_cvref_t<decltype(*well)>, X>) {
                const auto& cells = well->cells();
                Ax_local_.resize(cells.size());
                for (std::size_t i = 0; i < cells.size(); ++i) {
                    Ax_local_[i] = y[cells[i]];
                }
                well->applyFinish(Ax_local_);
                for (std::size_t i = 0; i < cells.size(); ++i) {
                    y[cells[i]] = Ax_local_[i];
                }
            } else {
                this->applySingleWell(x, y, well, well->cells());
            }
        }
    }

    void applyscaleaddFinish(field_type alpha, const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(applyscaleaddFinish);
        if (this->wellMod_.empty()) {
            return;
        }

        if (scaleAddRes_.size() != y.size()) {
            scaleAddRes_.resize(y.size());
        }

        scaleAddRes_ = 0.0;
        // scaleAddRes_  = - C D^-1 B x
        applyFinish(x, scaleAddRes_);
        // Ax = Ax + alpha * scaleAddRes_
        y.axpy(alpha, scaleAddRes_);
    }

    void applyscaleaddLocal(field_type alpha, const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(applyscaleaddLocal);
        if (this->wellMod_.empty()) {
            return;
        }

        if (scaleAddRes_.size() != y.size()) {
            scaleAddRes_.resize(y.size());
        }

        scaleAddRes_ = 0.0;
        for (const auto& well : this->wellMod_) {
            const auto& cells = well->cells();
            x_local_.resize(cells.size());
            Ax_local_.resize(cells.size());
            for (std::size_t i = 0; i < cells.size(); ++i) {
                x_local_[i] = x[cells[i]];
                Ax_local_[i] = 0.0;
            }
            // Ax_local = - C_k D^-1 B_k x_local
            if constexpr (TwoPhaseApplyWell<std::remove_cvref_t<decltype(*well)>, X>) {
                well->applyLocal(x_local_, Ax_local_);
            } else {
                well->apply(x_local_, Ax_local_);
            }
            for (std::size_t i = 0; i < cells.size(); ++i) {
                scaleAddRes_[cells[i]] += Ax_local_[i];
            }
        }
        y.axpy(alpha, scaleAddRes_);
    }

    /// Category for operator.
    /// This is somewhat tricky, I consider this operator sequential
    /// since (unlike WellModelMatrixAdapter) it does not do any
    /// projections etc. but only forwards the calls to the sequential
    /// well model.
    Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::sequential;
    }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                  const X& weights,
                                  const bool use_well_weights,
                                  const bool contract_d_diagonal) const override
    {
        OPM_TIMEBLOCK(addWellPressureEquations);
        wellMod_.addWellPressureEquations(jacobian, weights, use_well_weights, contract_d_diagonal);
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const override
    {
        OPM_TIMEBLOCK(addWellPressureEquationsStruct);
        wellMod_.addWellPressureEquationsStruct(jacobian);
    }

    int getNumberOfExtraEquations() const override
    {
        return wellMod_.numLocalWellsEnd();
    }

protected:
    const WellModel& wellMod_;

    template<class WellType, class ArrayType>
    void applySingleWell(const X& x, Y& y,
                         const WellType& well,
                         const ArrayType& cells) const
    {
        // Well equations B and C uses only the perforated cells, so need to apply on local vectors
        x_local_.resize(cells.size());
        Ax_local_.resize(cells.size());

        for (size_t i = 0; i < cells.size(); ++i) {
            x_local_[i] = x[cells[i]];
            Ax_local_[i] = y[cells[i]];
        }

        well->apply(x_local_, Ax_local_);

        for (size_t i = 0; i < cells.size(); ++i) {
            // only need to update Ax
            y[cells[i]] = Ax_local_[i];
        }

    }

    // These members are used to avoid reallocation.
    // Their state is not relevant between function calls, so they can
    // (and must) be mutable, as the functions using them are const.
    mutable X x_local_{};
    mutable Y Ax_local_{};
    mutable Y scaleAddRes_{};
};

template <class WellModel, class X, class Y>
class DomainWellModelAsLinearOperator : public WellModelAsLinearOperator<WellModel, X, Y>
{
public:
    using WBase = WellModelAsLinearOperator<WellModel, X, Y>;
    using WBase::WBase;  // inherit all constructors from the base class
    using field_type = typename WBase::field_type;
    using PressureMatrix = typename WBase::PressureMatrix;

    void setDomainIndex(int index) { domainIndex_ = index; }

    void apply(const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(apply);
        std::size_t well_index = 0;
        for (const auto& well : this->wellMod_) {
            if (this->wellMod_.well_domain().at(well->name()) == domainIndex_) {
                this->applySingleWell(x, y, well,
                                      this->wellMod_.well_local_cells()[well_index]);
            }
            ++well_index;
        }
    }

    // Only the wells of the domain are applied, in one phase.
    void applyStart(const X& /*x*/) const override {}

    void applyFinish(const X& x, Y& y) const override
    {
        apply(x, y);
    }

    void applyscaleaddFinish(field_type alpha, const X& x, Y& y) const override
    {
        if (this->scaleAddRes_.size() != y.size()) {
            this->scaleAddRes_.resize(y.size());
        }
        this->scaleAddRes_ = 0.0;
        apply(x, this->scaleAddRes_);
        y.axpy(alpha, this->scaleAddRes_);
    }

    void applyscaleaddLocal(field_type alpha, const X& x, Y& y) const override
    {
        this->applyscaleadd(alpha, x, y);
    }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                  const X& weights,
                                  const bool use_well_weights,
                                  const bool contract_d_diagonal) const override
    {
        OPM_TIMEBLOCK(addWellPressureEquations);
        this->wellMod_.addWellPressureEquationsDomain(jacobian,
                                                      weights,
                                                      use_well_weights,
                                                      contract_d_diagonal,
                                                      domainIndex_);
    }

private:
    int domainIndex_ = -1;
};

/*!
   \brief Adapter to combine a matrix and another linear operator into
   a combined linear operator.

   Adapts a matrix A plus another linear operator W (typically from
   wells) to the assembled linear operator interface by returning S
   from getmat() and making apply() and applyscaleadd() apply both A
   and W to the input vector. The adapter is for serial use only as the
   current parallel version in use is WellModelGhostLastMatrixAdapter.
 */
template<class M, class X, class Y>
class WellModelMatrixAdapter : public Dune::AssembledLinearOperator<M,X,Y>
{
public:
    using matrix_type = M;
    using domain_type = X;
    using range_type = Y;
    using field_type = typename X::field_type;
    using PressureMatrix = Dune::BCRSMatrix<MatrixBlock<field_type, 1, 1>>;

    Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::sequential;
    }

    //! constructor: just store a reference to a matrix
    WellModelMatrixAdapter (const M& A,
                            const LinearOperatorExtra<X, Y>& wellOper)
        : A_( A ), wellOper_( wellOper )
    {}

    void apply( const X& x, Y& y ) const override
    {
      OPM_TIMEBLOCK(apply);
      A_.mv(x, y);

      // add well model modification to y
      wellOper_.apply(x, y);
    }

    // y += \alpha * A * x
    void applyscaleadd (field_type alpha, const X& x, Y& y) const override
    {
      OPM_TIMEBLOCK(applyscaleadd);
      A_.usmv(alpha, x, y);

      // add scaled well model modification to y
      wellOper_.applyscaleadd(alpha, x, y);
    }

    //! \brief As applyscaleadd(), but distributed wells only use local perforations.
    void applyscaleaddLocalWells(field_type alpha, const X& x, Y& y) const
    {
      A_.usmv(alpha, x, y);
      wellOper_.applyscaleaddLocal(alpha, x, y);
    }

    const matrix_type& getmat() const override { return A_; }

    const LinearOperatorExtra<X, Y>& getwellOper() const { return wellOper_; }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                  const X& weights,
                                  const bool use_well_weights,
                                  const bool contract_d_diagonal) const
    {
        OPM_TIMEBLOCK(addWellPressureEquations);
        wellOper_.addWellPressureEquations(jacobian, weights, use_well_weights, contract_d_diagonal);
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const
    {
        OPM_TIMEBLOCK(addWellPressureEquations);
        wellOper_.addWellPressureEquationsStruct(jacobian);
    }

    int getNumberOfExtraEquations() const
    {
        return wellOper_.getNumberOfExtraEquations();
    }

protected:
    const matrix_type& A_ ;
    const LinearOperatorExtra<X, Y>& wellOper_;
};

/*!
   \brief Adapter to combine a matrix and another linear operator into
   a combined linear operator.

   This is similar to WellModelMatrixAdapter, with the difference that
   here we assume a parallel ordering of rows, where ghost rows are
   located after interior rows.
 */
template<class M, class X, class Y, bool overlapping >
class WellModelGhostLastMatrixAdapter : public Dune::AssembledLinearOperator<M,X,Y>
{
public:
    using matrix_type = M;
    using domain_type = X;
    using range_type = Y;
    using field_type = typename X::field_type;
    using PressureMatrix = Dune::BCRSMatrix<MatrixBlock<field_type, 1, 1>>;
#if HAVE_MPI
    using communication_type = Dune::OwnerOverlapCopyCommunication<int,int>;
#else
    using communication_type = Dune::Communication<int>;
#endif

    Dune::SolverCategory::Category category() const override
    {
        return overlapping ?
            Dune::SolverCategory::overlapping : Dune::SolverCategory::sequential;
    }

    //! constructor: just store a reference to a matrix
    WellModelGhostLastMatrixAdapter (const M& A,
                                     const LinearOperatorExtra<X, Y>& wellOper,
                                     const std::size_t interiorSize )
        : A_( A ), wellOper_( wellOper ), interiorSize_(interiorSize)
    {}

    void apply(const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(apply);
        // Start the well reductions first, so they overlap with the reservoir product
        wellOper_.applyStart(x);

        for (auto row = A_.begin(); row.index() < interiorSize_; ++row)
        {
            y[row.index()]=0;
            auto endc = (*row).end();
            for (auto col = (*row).begin(); col != endc; ++col)
                (*col).umv(x[col.index()], y[row.index()]);
        }

        // add well model modification to y
        wellOper_.applyFinish(x, y);

        ghostLastProject(y);
    }

    // y += \alpha * A * x
    void applyscaleadd (field_type alpha, const X& x, Y& y) const override
    {
        OPM_TIMEBLOCK(applyscaleadd);
        // Start the well reductions first, so they overlap with the reservoir product
        wellOper_.applyStart(x);

        reservoirScaleAdd(alpha, x, y);

        // add scaled well model modification to y
        wellOper_.applyscaleaddFinish(alpha, x, y);

        ghostLastProject(y);
    }

    //! \brief As applyscaleadd(), but distributed wells only use the perforations on this
    //! process (no communication). Used for residual updates inside preconditioners.
    void applyscaleaddLocalWells(field_type alpha, const X& x, Y& y) const
    {
        OPM_TIMEBLOCK(applyscaleaddLocalWells);
        reservoirScaleAdd(alpha, x, y);
        wellOper_.applyscaleaddLocal(alpha, x, y);
        ghostLastProject(y);
    }

    const matrix_type& getmat() const override { return A_; }

    const LinearOperatorExtra<X, Y>& getwellOper() const { return wellOper_; }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                  const X& weights,
                                  const bool use_well_weights,
                                  const bool contract_d_diagonal) const
    {
        OPM_TIMEBLOCK(addWellPressureEquations);
        wellOper_.addWellPressureEquations(jacobian, weights, use_well_weights, contract_d_diagonal);
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const
    {
        OPM_TIMEBLOCK(addWellPressureEquationsStruct);
        wellOper_.addWellPressureEquationsStruct(jacobian);
    }

    int getNumberOfExtraEquations() const
    {
        return wellOper_.getNumberOfExtraEquations();
    }

protected:
    void reservoirScaleAdd(field_type alpha, const X& x, Y& y) const
    {
        for (auto row = A_.begin(); row.index() < interiorSize_; ++row)
        {
            auto endc = (*row).end();
            for (auto col = (*row).begin(); col != endc; ++col)
                (*col).usmv(alpha, x[col.index()], y[row.index()]);
        }
    }

    void ghostLastProject(Y& y) const
    {
        std::size_t end = y.size();
        for (std::size_t i = interiorSize_; i < end; ++i)
            y[i] = 0;
    }

    const matrix_type& A_ ;
    const LinearOperatorExtra<X, Y>& wellOper_;
    std::size_t interiorSize_;
};

//! \brief Operators that can be applied with distributed wells using only local perforations.
template<class FO>
concept HasLocalWellsApply = requires(const FO& op,
                                      typename FO::field_type alpha,
                                      const typename FO::domain_type& x,
                                      typename FO::range_type& y)
{
    op.applyscaleaddLocalWells(alpha, x, y);
};

/*!
   \brief View of a well-aware operator where distributed wells are applied using only
   the perforations on this process, i.e. with the rank-local Schur complement
   \f$ A_k - C_k^T D^{-1} B_k \f$. It needs no well communication.

   This is not the system operator, so it is only meant for use inside a (fixed, linear)
   preconditioner, e.g. for the residual updates of the CPR smoothing steps.
 */
template<class FO>
class LocalWellsOperatorView
    : public Dune::LinearOperator<typename FO::domain_type, typename FO::range_type>
{
public:
    using domain_type = typename FO::domain_type;
    using range_type = typename FO::range_type;
    using field_type = typename FO::field_type;

    explicit LocalWellsOperatorView(const FO& op)
        : op_(op)
    {}

    void apply(const domain_type& x, range_type& y) const override
    {
        y = 0;
        op_.applyscaleaddLocalWells(field_type(1), x, y);
    }

    void applyscaleadd(field_type alpha, const domain_type& x, range_type& y) const override
    {
        op_.applyscaleaddLocalWells(alpha, x, y);
    }

    Dune::SolverCategory::Category category() const override
    {
        return op_.category();
    }

private:
    const FO& op_;
};

/*!
   \brief Dune linear operator that assumes ghost rows are ordered after
   interior rows. Avoids some computations because of this.

   This is similar to WellModelGhostLastMatrixAdapter, with the difference that
   here we do not have a well model, and also do calcilate the interiorSize
   using the parallel index set. Created for use in AMG/CPR smoothers.
 */
template<class M, class X, class Y, class C>
class GhostLastMatrixAdapter : public Dune::AssembledLinearOperator<M,X,Y>
{
public:
    typedef M matrix_type;
    typedef X domain_type;
    typedef Y range_type;
    typedef typename X::field_type field_type;


    typedef C communication_type;

    Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::overlapping;
    }

    //! constructor: just store a reference to a matrix
    GhostLastMatrixAdapter (const M& A,
                            const communication_type& comm)
        : A_( Dune::stackobject_to_shared_ptr(A) ), comm_(comm)
    {
        interiorSize_ = setInteriorSize(comm_);
    }

    GhostLastMatrixAdapter (const std::shared_ptr<M> A,
                            const communication_type& comm)
        : A_( A ), comm_(comm)
    {
        interiorSize_ = setInteriorSize(comm_);
    }

    void apply( const X& x, Y& y ) const override
    {
        for (auto row = A_->begin(); row.index() < interiorSize_; ++row)
        {
            y[row.index()]=0;
            auto endc = (*row).end();
            for (auto col = (*row).begin(); col != endc; ++col)
                (*col).umv(x[col.index()], y[row.index()]);
        }

        ghostLastProject( y );
    }

    // y += \alpha * A * x
    void applyscaleadd (field_type alpha, const X& x, Y& y) const override
    {
        for (auto row = A_->begin(); row.index() < interiorSize_; ++row)
        {
            auto endc = (*row).end();
            for (auto col = (*row).begin(); col != endc; ++col)
                (*col).usmv(alpha, x[col.index()], y[row.index()]);
        }

        ghostLastProject( y );
    }

    const matrix_type& getmat() const override { return *A_; }

    size_t getInteriorSize() const { return interiorSize_;}

private:
    void ghostLastProject(Y& y) const
    {
        size_t end = y.size();
        for (size_t i = interiorSize_; i < end; ++i)
            y[i] = 0; //project to interiorsize, i.e. ignore ghost
    }

    size_t setInteriorSize(const communication_type& comm) const
    {
        auto indexSet = comm.indexSet();
        if (indexSet.size() == 0)
            return 0;

        size_t is = 0;
        // Loop over index set
        for (auto idx = indexSet.begin(); idx!=indexSet.end(); ++idx) {
            //Only take "owner" indices
            if (idx->local().attribute()==1) {
                //get local index
                auto loc = idx->local().local();
                // if loc is higher than "old interior size", update it
                if (loc > is) {
                    is = loc;
                }
            }
        }
        return is + 1; //size is plus 1 since we start at 0, except when indexSet is empty
    }
    const std::shared_ptr<const matrix_type> A_ ;
    const communication_type&  comm_;
    size_t interiorSize_;
};

} // namespace Opm

namespace Dune {
namespace Amg {

template<class M, class X, class Y, class C>
class ConstructionTraits<Opm::GhostLastMatrixAdapter<M,X,Y,C> >
{
public:
    typedef ParallelOperatorArgs<M,C> Arguments;

    static inline std::shared_ptr<Opm::GhostLastMatrixAdapter<M,X,Y,C>> construct(const Arguments& args)
    {
        return std::make_shared<Opm::GhostLastMatrixAdapter<M,X,Y,C>>
            (args.matrix_, args.comm_);
    }
};

} // end namespace Amg
} // end namespace Dune

#endif // OPM_WELLOPERATORS_HEADER_INCLUDED
