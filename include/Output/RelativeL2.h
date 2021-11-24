#ifndef OUTPUT_RELATIVE_L2_H_
#define OUTPUT_RELATIVE_L2_H_

#include <Eigen/Core>
#include <stdexcept>
#include "../Config.h"

namespace MiniDNN
{


///
/// \ingroup Outputs
///
/// Relative L2 output layer
/// The Num and Denom are used to constructed epsilon with epsilon = (MiniDNN::Scalar) Num / Denom
///
template<std::intmax_t Num, std::intmax_t Denom>
class RelativeL2: public Output
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Matrix m_imm;   // (sg( ||yhat||^2 ) + epsilon) for each observation
        Matrix m_din;   // Derivative of the input of this layer.
        // Note that input of this layer is also the output of previous layer

        const MiniDNN::Scalar epsilon = (MiniDNN::Scalar)Num / Denom;

    public:
        void evaluate(const Matrix& prev_layer_data, const Matrix& target)
        {
            // Check dimension
            // Number of observations
            const int nobs = prev_layer_data.cols();

            // Number of the variables
            const int nvar = prev_layer_data.rows();

            if ((target.cols() != nobs) || (target.rows() != nvar))
            {
                throw std::invalid_argument("[class RelativeL2]: Target data have incorrect dimension");
            }

            // Compute the derivative of the input of this layer
            // L = 0.5 * ||(yhat - y)||^2 / (sg( ||yhat||^2 ) + epsilon)
            // -> where sg() means no gradient is propagated back here

            // d(L) / d(in) = (yhat - y) / (sg( ||yhat||^2 ) + epsilon)
            m_din.resize(nvar, nobs);
            m_imm.resize(1, nobs);

            for (int i = 0; i < nobs; i++) {
                m_imm(0, i) = (prev_layer_data.col(i).squaredNorm() + epsilon);
                m_din.col(i).noalias() = (prev_layer_data.col(i) - target.col(i)) / m_imm(0, i);
            }
        }

        const Matrix& backprop_data() const
        {
            return m_din;
        }

        // Average loss of all observations
        Scalar loss() const
        {
            // L = 0.5 * ||(yhat - y)||^2 / (sg( ||yhat||^2 ) + epsilon)
            Scalar loss = 0;
            const int nobs = m_din.cols();

            for (int i = 0; i < nobs; i++) {
                // (yhat - y) = m_din.col(i) * m_imm.col(i)
                // ||(yhat - y)||^2 / (sg( ||yhat||^2 ) + epsilon) = (m_din.col(i) * m_imm.col(i)).squaredNorm() / m_imm.col(i)
                //                                                 = (m_din.col(i)).squaredNorm() * m_imm.col(i)
                loss += (m_din.col(i)).squaredNorm() * m_imm(0, i); 
            }
            return (Scalar(0.5) * loss) / nobs;
        }

        std::string output_type() const
        {
            return "RelativeL2";
        }
};


} // namespace MiniDNN


#endif /* OUTPUT_RELATIVE_L2_H_ */
