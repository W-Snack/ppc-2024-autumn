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
  boost::mpi::communicator world;
  char* input_;
  int sentence_count;
  int local_sentence_count;
  std::vector<char> local_input;
};

}  // namespace fomin_v_sentence_count
