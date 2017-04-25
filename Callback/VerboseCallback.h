#ifndef CALLBACK_VERBOSECALLBACK_H_
#define CALLBACK_VERBOSECALLBACK_H_

#include <Eigen/Core>
#include <iostream>
#include "../Config.h"
#include "../Callback.h"
#include "../Network.h"

class VerboseCallback: public Callback
{
public:
    void post_training_batch(const Network* net, const Matrix& x, const Matrix& y)
    {
        Scalar loss = net->loss(y);
        std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = " << loss << std::endl;
    }

    void post_training_batch(const Network* net, const Matrix& x, const IntegerVector& y)
    {
        Scalar loss = net->loss(y);
        std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = " << loss << std::endl;
    }
};


#endif /* CALLBACK_VERBOSECALLBACK_H_ */
