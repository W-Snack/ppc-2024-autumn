#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

TEST(fomin_v_generalized_scatter, ScatterIntegers) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int data_size = size * 10;
  int* sendbuf = nullptr;
  auto* recvbuf = new int[10];

  if (rank == root) {
    sendbuf = new int[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = i;
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_INT, recvbuf, 10, MPI_INT, root,
                                                   MPI_COMM_WORLD);

  int expected[10];
  for (int i = 0; i < 10; ++i) {
    expected[i] = rank * 10 + i;
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, ScatterFloats) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int data_size = size * 10;
  float* sendbuf = nullptr;
  auto* recvbuf = new float[10];

  if (rank == root) {
    sendbuf = new float[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = static_cast<float>(i);
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_FLOAT, recvbuf, 10, MPI_FLOAT, root,
                                                   MPI_COMM_WORLD);

  float expected[10];
  for (int i = 0; i < 10; ++i) {
    expected[i] = static_cast<float>(rank * 10 + i);
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, ScatterDoubles) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int data_size = size * 10;
  double* sendbuf = nullptr;
  auto* recvbuf = new double[10];

  if (rank == root) {
    sendbuf = new double[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = static_cast<double>(i);
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_DOUBLE, recvbuf, 10, MPI_DOUBLE, root,
                                                   MPI_COMM_WORLD);

  double expected[10];
  for (int i = 0; i < 10; ++i) {
    expected[i] = static_cast<double>(rank * 10 + i);
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, NonPowerOfTwoProcesses) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int data_size = size * 10;
  int* sendbuf = nullptr;
  auto* recvbuf = new int[10];

  if (rank == root) {
    sendbuf = new int[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = i;
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_INT, recvbuf, 10, MPI_INT, root,
                                                   MPI_COMM_WORLD);

  int expected[10];
  for (int i = 0; i < 10; ++i) {
    expected[i] = rank * 10 + i;
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, ZeroElementsScatter) {
  boost::mpi::communicator world;
  int root = 0;
  const int data_size = 0;
  int* sendbuf = nullptr;
  int* recvbuf = nullptr;
  int result = fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_INT, recvbuf, 0, MPI_INT, root,
                                                                MPI_COMM_WORLD);

  // Check that the function returns MPI_SUCCESS
  EXPECT_EQ(result, MPI_SUCCESS);
}