#include "seq/fomin_v_sentence_count/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

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
      if (!in_sentence) {
        sentence_count++;
        in_sentence = true;
      }
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
