#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> fomin_v_generalized_scatter::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

int fomin_v_generalized_scatter::generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                                     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                                     MPI_Comm comm) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int datatype_size;
  MPI_Type_size(sendtype, &datatype_size);

  // Calculate subtree sizes
  auto* subtree_sizes = new int[size];
  for (int i = size - 1; i >= 0; --i) {
    subtree_sizes[i] = 1;
    if (2 * i + 1 < size) subtree_sizes[i] += subtree_sizes[2 * i + 1];
    if (2 * i + 2 < size) subtree_sizes[i] += subtree_sizes[2 * i + 2];
  }

  if (rank == root && sendcount != subtree_sizes[root] * recvcount) {
    delete[] subtree_sizes;
    return MPI_ERR_COUNT;
  }

  int parent = (rank == root) ? MPI_PROC_NULL : (rank - 1) / 2;
  int left_child = 2 * rank + 1;
  int right_child = 2 * rank + 2;

  char* temp_buffer = nullptr;
  if (rank != root) {
    // Allocate buffer for receiving data from the parent
    temp_buffer = new char[subtree_sizes[rank] * recvcount * datatype_size];
  }

  if (rank == root) {
    const char* send_ptr = static_cast<const char*>(sendbuf);

    // Copy root's data
    memcpy(recvbuf, send_ptr, recvcount * datatype_size);

    // Send data to left child
    if (left_child < size) {
      int left_offset = recvcount * datatype_size;
      int left_data_size = subtree_sizes[left_child] * recvcount * datatype_size;
      MPI_Send(send_ptr + left_offset, left_data_size / datatype_size, sendtype, left_child, 0, comm);
    }

    // Send data to right child
    if (right_child < size) {
      int right_offset = (recvcount + subtree_sizes[left_child] * recvcount) * datatype_size;
      int right_data_size = subtree_sizes[right_child] * recvcount * datatype_size;
      MPI_Send(send_ptr + right_offset, right_data_size / datatype_size, sendtype, right_child, 0, comm);
    }
  } else {
    // Receive data from parent
    MPI_Status status;
    MPI_Recv(temp_buffer, subtree_sizes[rank] * recvcount, sendtype, parent, 0, comm, &status);

    // Copy data for the current process
    memcpy(recvbuf, temp_buffer, recvcount * datatype_size);

    // Forward data to left child
    if (left_child < size) {
      int left_data_size = subtree_sizes[left_child] * recvcount * datatype_size;
      MPI_Send(temp_buffer + recvcount * datatype_size, left_data_size / datatype_size, sendtype, left_child, 0, comm);
    }

    // Forward data to right child
    if (right_child < size) {
      int offset = (recvcount + subtree_sizes[left_child] * recvcount) * datatype_size;
      int right_data_size = subtree_sizes[right_child] * recvcount * datatype_size;
      MPI_Send(temp_buffer + offset, right_data_size / datatype_size, sendtype, right_child, 0, comm);
    }
  }

  if (temp_buffer != nullptr) {
    delete[] temp_buffer;
  }
  delete[] subtree_sizes;
  return MPI_SUCCESS;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::pre_processing() {
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[0] % taskData->outputs_count[0] == 0;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::run() {
  internal_order_test();
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int root = 0;

  int sendcount = taskData->inputs_count[0];
  int recvcount = sendcount / size;

  if (rank == root) {
    int err = generalized_scatter(taskData->inputs[0], sendcount, MPI_INT, taskData->outputs[0], recvcount, MPI_INT,
                                  root, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
      // std::cerr << "Error in generalized_scatter on root process." << std::endl;
      return false;
    }
  } else {
    int err = generalized_scatter(nullptr, 0, MPI_INT, taskData->outputs[0], recvcount, MPI_INT, root, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
      // std::cerr << "Error in generalized_scatter on process " << rank << std::endl;
      return false;
    }
  }

  return true;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}