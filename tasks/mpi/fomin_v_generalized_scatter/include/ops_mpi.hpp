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

namespace fomin_v_generalized_scatter {

// Declaration of getRandomVector function
std::vector<int> getRandomVector(int sz);
// Declaration of generalized_scatter function
int generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                        MPI_Datatype recvtype, int root, MPI_Comm comm);

// Declaration of GeneralizedScatterTestParallel class
class GeneralizedScatterTestParallel : public ppc::core::Task {
 public:
  explicit GeneralizedScatterTestParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{};
  std::string local_input_{};
  int res{};
  boost::mpi::communicator world;
};

}  // namespace fomin_v_generalized_scatter