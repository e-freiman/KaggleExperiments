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

using Idx = Row<uword>;

int main()
{
  constexpr uword MAX_N = 100e10;
  constexpr double RATIO = 0.8;
  constexpr int H1 = 600;
  constexpr int H2 = 100;
  constexpr int ITERATIONS = 100000;
  constexpr int BATCH_SIZE = 50;
  constexpr double STEP_SIZE = 1e-3; 
  
  std::cout << "Reading data ..." << std::endl;
  
  mat dataset;
  data::Load("../../MNIST-Data/train.csv", dataset, true);
  const uword N = std::min(MAX_N, dataset.n_cols);
  const uword splitter = N * 0.8;
  
  const Idx featureIdx = regspace<Idx>(1, dataset.n_rows - 1);
  const Idx labelIdx = regspace<Idx>(0, 0);

  const Idx idx = shuffle(regspace<Idx>(0,  N - 1));
  const Idx trainIdx = idx.subvec(0, splitter - 1);
  const Idx testIdx = idx.subvec(splitter, idx.n_elem - 1);
  
  const mat trainX = dataset.submat(featureIdx, trainIdx);
  const mat testX = dataset.submat(featureIdx, testIdx);

  const mat tempTrainY = dataset.submat(labelIdx, trainIdx);
  const mat tempTestY = dataset.submat(labelIdx, testIdx);

  mat trainY(1, tempTrainY.n_cols);
  for (size_t j = 0; j < tempTrainY.n_cols; ++j)
  {
    trainY(0, j) = tempTrainY(0, j) + 1;
  }
  
  mat testY(1, tempTestY.n_cols);
  for (size_t j = 0; j < tempTestY.n_cols; ++j)
  {
    testY(0, j) = tempTestY(0, j) + 1;
  }
  
  FFN<> model;

  model.Add<Linear<> >(trainX.n_rows, H1);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(H1, H2);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(H2, 10);
  model.Add<LogSoftMax<> >();

  std::cout << "Training ..." << std::endl;  
  Adam opt(STEP_SIZE, BATCH_SIZE, 0.9, 0.999, 1e-8, ITERATIONS, 1e-8, true);
  
  model.Train(trainX, trainY, opt);

  std::cout << "Predicting ..." << std::endl;
  mat tempPredY;
  model.Predict(testX, tempPredY);
  
  mat pred = arma::zeros<mat>(1, tempPredY.n_cols);

  for (size_t j = 0; j < tempPredY.n_cols; ++j)
  {
    pred(j) = arma::as_scalar(arma::find(
        arma::max(tempPredY.col(j)) == tempPredY.col(j), 1)) + 1;
  }

  std::cout << "Checking ..." << std::endl;
  
  size_t success = 0;
  for (size_t j = 0; j < testY.n_cols; j++) {
    if (std::round(pred(j)) == std::round(testY(j))) {
      ++success;
    }  
  }
  
  double accuracy = (double)success / (double)testY.n_cols * 100.0;

  std::cout << "Finished" << std::endl;
  std::cout << "Accuracy "<< accuracy << "%" << std::endl;
}