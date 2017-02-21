/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <cstring>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void construct_net(network<sequential>& nn, core::backend_t backend_type) {
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
  // clang-format off
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
// clang-format on
#undef O
#undef X

  // construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  nn << convolutional_layer<tan_h>(32, 32, 5, 1,
                                   6,  // C1, 1@32x32-in, 6@28x28-out
                                   padding::valid, true, 1, 1, backend_type)
     << average_pooling_layer<tan_h>(28, 28, 6,
                                     2)  // S2, 6@28x28-in, 6@14x14-out
     << convolutional_layer<tan_h>(14, 14, 5, 6,
                                   16,  // C3, 6@14x14-in, 16@10x10-out
                                   connection_table(tbl, 6, 16), padding::valid,
                                   true, 1, 1, backend_type)
     << average_pooling_layer<tan_h>(10, 10, 16,
                                     2)  // S4, 16@10x10-in, 16@5x5-out
     << convolutional_layer<tan_h>(5, 5, 5, 16,
                                   120,  // C5, 16@5x5-in, 120@1x1-out
                                   padding::valid, true, 1, 1, backend_type)
    //  << fully_connected_layer<softmax>(120, 10,  // F6, 120-in, 10-out
     << fully_connected_layer<tan_h>(120, 10,  // F6, 120-in, 10-out
                                     true, backend_type);
}

static void train_lenet(const std::string& data_dir_path,
                        int minibatch_size,
                        int num_epochs,
                        core::backend_t backend_type) {
  // specify loss-function and learning strategy
  network<sequential> nn;
  adagrad optimizer;

  construct_net(nn, backend_type);

  std::cout << "load models..." << std::endl;

  // load MNIST dataset
  std::vector<label_t> train_labels, test_labels;
  std::vector<vec_t> train_images, test_images;

  parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
  parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images,
                     -1.0, 1.0, 2, 2);
  parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
  parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images,
                     -1.0, 1.0, 2, 2);

  std::cout << "start training" << std::endl;
  std::cout << "minibatch_size\t: " << minibatch_size << std::endl;
  std::cout << "num_epochs\t: " << num_epochs << std::endl;
  std::cout << "backend\t: " << backend_type << std::endl;

  progress_display disp(static_cast<unsigned long>(train_images.size()));
  timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size)));

  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << t.elapsed() << "s elapsed." << std::endl;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(static_cast<unsigned long>(train_images.size()));
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

  // training
  nn.train<mse>(optimizer, train_images, train_labels, minibatch_size,
                num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);

  // save network model & trained weights
  nn.save("LeNet-model");
}

static core::backend_t parse_backend_name(const char *name) {
  static const char *names[] = {
    "internal",
    "nnpack",
    "libdnn",
    "avx",
    "opencl",
  };
  for (size_t i=0; i<sizeof(names)/sizeof(names[0]); ++i) {
    if (strcasecmp(name, names[i]) == 0) {
      return (core::backend_t)i;
    }
  }
  return core::default_engine();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage : " << argv[0] << " path_to_data (example:../data)"
              << std::endl;
    return -1;
  }
  const char *dir = argv[1];
  int minibatch_size = (argc < 3) ? 16 : atoi(argv[2]);
  int num_epochs = (argc < 4) ? 30 : atoi(argv[3]);
  core::backend_t backend_type = (argc < 5) ? core::default_engine() : parse_backend_name(argv[4]);
  train_lenet(dir, minibatch_size, num_epochs, backend_type);
  return 0;
}

