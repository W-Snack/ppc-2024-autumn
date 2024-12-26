#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_sentence_count {

class SentenceCountSequential : public ppc::core::Task {
 public:
  explicit SentenceCountSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  char *input_;
  int sentence_count;
};

class SentenceCountParallel : public ppc::core::Task {
 public:
  explicit SentenceCountParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
 private:
  std::vector<char> input_vec;
  std::vector<char> local_input_vec;
  int input_size{};
  int portion_size{};
  int local_sentence_count{};
  int total_sentence_count{};
  char *input_ptr{};
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace fomin_v_sentence_count