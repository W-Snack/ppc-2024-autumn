#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstring>
#include <vector>

using namespace std::chrono_literals;

bool fomin_v_sentence_count::SentenceCountParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs[0] == nullptr) {
      return false;
    }
    input_ = reinterpret_cast<char *>(taskData->inputs[0]);
  }

  int input_size;
  if (world.rank() == 0) {
    input_size = static_cast<int>(strlen(input_));
  }
  broadcast(world, input_size, 0);

  int chunk_size = input_size / world.size();
  local_input.resize(chunk_size);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      int start = proc * chunk_size;
      world.send(proc, input_ + start, chunk_size, 0);
    }
  } else {
    world.recv(0, local_input.data(), chunk_size, 0);
  }

  local_sentence_count = 0;
  return true;
}

bool fomin_v_sentence_count::SentenceCountParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool fomin_v_sentence_count::SentenceCountParallel::run() {
  internal_order_test();

  for (size_t i = 0; i < local_input.size(); ++i) {
    if (local_input[i] == '.' || local_input[i] == '!' || local_input[i] == '?') {
      local_sentence_count++;
    }
  }

  reduce(world, local_sentence_count, sentence_count, std::plus(), 0);

  return true;
}

bool fomin_v_sentence_count::SentenceCountParallel::post_processing() {
  internal_order_test();

  // Проверка на nullptr
  if (taskData->outputs[0] == nullptr) {
    return false;
  }

  int total_sentence_count = 0;
  reduce(world, local_sentence_count, total_sentence_count, std::plus<int>(), 0);

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
  bool in_sentence = false;
  for (int i = 0; input_[i] != '\0'; ++i) {
    if (input_[i] == '.' || input_[i] == '!' || input_[i] == '?') {
      if (in_sentence) {
        sentence_count++;
      }
      in_sentence = false;
    } else if (isspace(input_[i])) {
      in_sentence = false;
    } else {
      in_sentence = true;
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
