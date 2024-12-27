#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstring>
#include <vector>

using namespace std::chrono_literals;

bool fomin_v_sentence_count::SentenceCountParallel::pre_processing() {
  internal_order_test();
  int world_size = world.size();
  int world_rank = world.rank();

  if (world_rank == 0) {
    input_ptr = reinterpret_cast<char *>(taskData->inputs[0]);
    input_size = taskData->inputs_count[0];
    input_vec.assign(input_ptr, input_ptr + input_size);
  }

  boost::mpi::broadcast(world, input_size, 0);

  portion_size = input_size / world_size;
  int remainder = input_size % world_size;
  if (world_rank < remainder) {
    portion_size++;
  }

  local_input_vec.resize(portion_size + 1);
  local_input_vec[portion_size] = '\0';

  if (world_rank == 0) {
    boost::mpi::scatter(world, input_vec, local_input_vec.data(), portion_size, 0);
  } else {
    boost::mpi::scatter(world, std::vector<char>(), local_input_vec.data(), portion_size, 0);
  }

  return true;
}

bool fomin_v_sentence_count::SentenceCountParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == static_cast<unsigned int>(input_size) && taskData->outputs_count[0] == 1;
  } else {
    return taskData->inputs_count[0] == static_cast<unsigned int>(input_size);
  }
}

bool fomin_v_sentence_count::SentenceCountParallel::run() {
  internal_order_test();
  local_sentence_count = 0;
  bool potential_sentence_end = false;
  
  for (int i = 0; i < portion_size; ++i) {
    if (ispunct(local_input_vec[i]) && 
        (local_input_vec[i] == '.' || local_input_vec[i] == '!' || local_input_vec[i] == '?')) {
      // Проверяем, что это конец предложения
      if (i == portion_size - 1) {
        potential_sentence_end = true;
      } else if (isspace(local_input_vec[i + 1]) || local_input_vec[i + 1] == '\0') {
        local_sentence_count++;
      }
    }
  }
  
  int send_potential_end = potential_sentence_end ? 1 : 0;
  int recv_potential_end = 0;
  boost::mpi::allreduce(world, &send_potential_end, 1, &recv_potential_end, std::plus<int>());
  
  return true;
}

bool fomin_v_sentence_count::SentenceCountParallel::post_processing() {
  internal_order_test();
  int total_sentence_count = 0;
  boost::mpi::reduce(world, local_sentence_count, total_sentence_count, std::plus<int>(), 0);
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = total_sentence_count;
  }

  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::pre_processing() {
  internal_order_test();
  // Получаем входную строку
  input_ = reinterpret_cast<char *>(taskData->inputs[0]);
  sentence_count = 0;
  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::validation() {
  internal_order_test();
  // Проверяем, что входные данные содержат строку
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool fomin_v_sentence_count::SentenceCountSequential::run() {
  internal_order_test();
  // Подсчитываем количество предложений
  for (int i = 0; input_[i] != '\0'; ++i) {
    if (ispunct(input_[i]) && (input_[i] == '.' || input_[i] == '!' || input_[i] == '?')) {
      sentence_count++;
    }
  }
  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::post_processing() {
  internal_order_test();
  // Записываем результат в выходные данные
  reinterpret_cast<int *>(taskData->outputs[0])[0] = sentence_count;
  return true;
}
