#include <iostream>

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

using namespace arma;

// Returns accuracy (percentage of correct answers)
double accuracy(const mat& predOut, const mat& realY)
{
  // Variable for storing correct labels
  mat pred = arma::zeros<mat>(1, predOut.n_cols);
  
  // predOut contains 10 rows, each corresponds a class and contains
  // the log of probability of datapont to be in that class. Class of 
  // a datapont is choosed to be the one with maximum of log of probability.
  for (int j = 0; j < predOut.n_cols; ++j)
  {
    pred(j) = arma::as_scalar(arma::find(
        arma::max(predOut.col(j)) == predOut.col(j), 1)) + 1;
  }
  
  // Calculating how many predicted classes are coincide with proper
  // labels.
  int success = 0;
  for (int j = 0; j < realY.n_cols; j++) {
    if (std::round(pred(j)) == std::round(realY(j))) {
      ++success;
    }  
  }
  
  // Calcualting percentage of correct classifications
  return (double)success / (double)realY.n_cols * 100.0;
}

int main()
{
  // Dataset is randomly split into training 
  // and test parts with followig ratio.
  constexpr double RATIO = 0.8;
  // The number of neurons in the first layer.
  constexpr int H1 = 600;
  // The number of neurons in the second layer.
  constexpr int H2 = 600;
  
  // The solution is done in several approaches (CYCLES), each approach 
  // uses previous results as starting point and have different optimizer 
  // options (here the step size is different).
  
  // Number of iteration per cycle. 
  constexpr int ITERATIONS_PER_CYCLE = 5000;
  
  // Number of cycles.
  constexpr int CYCLES = 50;
  
  // Initial step size of an optimizer.
  constexpr double STEP_BEGIN = 1e-2;
  
  // Final step size of an optimizer. Between those two points step size is
  // vary linearly.
  constexpr double STEP_END = 1e-3;
  
  std::cout << "Reading data ..." << std::endl;
  
  // Labeled dataset that contins data for training is loaded from csv file,
  // rows represent features, columns represent data points.
  mat dataset;
  data::Load("../../MNIST-Data/train.csv", dataset, true);
  
  const int N = dataset.n_cols;
  
  // The index that splites dataset on two parts: for training and testing.
  const int SPLITTER = N * 0.8;
  
  // Generating row of indexes of features (rows of dataset from 1 to 
  // dataset.n_rows - 1).
  const Row<uword> featureIdx = regspace<Row<uword>>(1, dataset.n_rows - 1);

  // Generating shuffled row of indexes of datapoits.
  const Row<uword> idx = shuffle(regspace<Row<uword>>(0,  N - 1));
  
  // Getting indexes of trining subset of data points.
  const Row<uword> trainIdx = idx.subvec(0, SPLITTER - 1);
  // Getting indexes of testing subset of data points.
  const Row<uword> testIdx = idx.subvec(SPLITTER, idx.n_elem - 1);
  
  // Getting training dataset with features (subset of original dataset
  // with selectes for training indexes)
  const mat trainX = dataset.submat(featureIdx, trainIdx);
  // Getting testing dataset with features (subset of original dataset
  // with selectes for testing indexes).
  const mat testX = dataset.submat(featureIdx, testIdx);

  // According to NegativeLogLikelihood output layer of NN, labels should
  // specify class of a datapoint and be in the interval from 1 to 
  // number of classes (in this case from 1 to 10).
  
  // Creating labels for training.
  mat trainY(1, trainIdx.n_cols);
  for (int j = 0; j < trainIdx.n_cols; ++j)
  {
    trainY(0, j) = dataset(0, trainIdx(j)) + 1;
  }
  
  // Creating labels for testing.
  mat testY(1, tempTestY.n_cols);
  for (int j = 0; j < testIdx.n_cols; ++j)
  {
    testY(0, j) = dataset(0, testIdx(j)) + 1;
  }
  
  // Specifing the NN model. NegativeLogLikelihood is the output layer that
  // is used for classification problem. RandomInitialization means that 
  // initial weights in neurons are going to be generated randomly
  // in the interval from -1 to 1.
  FFN<NegativeLogLikelihood<>, RandomInitialization> model;
  // This is intermediate layer that is needed for connection between input
  // data and sigmoid layer. Parameters specify the number of input features
  // and number of neurons in the next layer.
  model.Add<Linear<> >(trainX.n_rows, H1);
  // The first sigmoid layer.
  model.Add<SigmoidLayer<> >();
  // Intermediate layer between sigmoid layers.
  model.Add<Linear<> >(H1, H2);
  // The second sigmoid layer.
  model.Add<SigmoidLayer<> >();
  // Dropout layer for regularization. First parameter - the probability of
  // setting a specific value to 0.
  model.Add<Dropout<> >(0.3, true);
  // Intermediate layer.
  model.Add<Linear<> >(H2, 10);
  // LogSoftMax layer is used together with NegativeLogLikelihood for mapping
  // output values to log of probabilities of being a specific class.
  model.Add<LogSoftMax<> >();
  
  std::cout << "Training ..." << std::endl;  
  
  // Cycles for minitoring a solution and applaying different step size of 
  // the Adam optimizer.
  for (int i = 0; i <= CYCLES; i++) {
    
    // Calculating a step size as linearly distributed between specifed 
    // STEP_BEGIN and STEP_END values.
    double stepRatio = (double)i / (double)CYCLES;
    double step = STEP_BEGIN + stepRatio * (STEP_END - STEP_BEGIN);
    
    // Setting parameters of Adam optimizer.
    Adam opt(step, 50, 0.9, 0.999, 1e-8, ITERATIONS_PER_CYCLE, 1e-8, true);  
    
    // Train neural network. If this is the first iteration, weights are 
    // random, using current values as starting point otherwise.
    model.Train(trainX, trainY, opt);  
    
    mat predOut;
    // Getting predictions on training dataset.
    model.Predict(trainX, predOut);
    // Calculating acccuracy on training dataset.
    double trainAccuracy = accuracy(predOut, trainY);
    // Getting predictions on test dataset.
    model.Predict(testX, predOut);
    // Calculating acccuracy on test dataset.
    double testAccuracy = accuracy(predOut, testY);

    std::cout << i << ", step = " << step << ", accuracy"
      " train = "<< trainAccuracy << "%," << 
      " test = "<< testAccuracy << "%" <<  std::endl;
  }
  
  std::cout << "Finished" << std::endl;
}