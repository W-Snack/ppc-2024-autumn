#include <gtest/gtest.h>

#include <ppc/core/perf.hpp>

#include "seq/fomin_v_sentence_count/include/ops_seq.hpp"

TEST(sequential_sentence_count_perf_test, test_pipeline_run) {
  // Входная строка с несколькими предложениями
  std::string input = "Hello world! How are you? I'm fine. This is a test. Another sentence!";
  std::vector<int> out(1, 0);

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создаем задачу
  auto sentenceCountTask = std::make_shared<fomin_v_sentence_count::SentenceCountSequential>(taskDataSeq);

  // Создаем атрибуты производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Количество запусков
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверяем результат
  ASSERT_EQ(5, out[0]);  // Ожидаемое количество предложений
}

TEST(sequential_sentence_count_perf_test, test_task_run) {
  // Входная строка с несколькими предложениями
  std::string input = "Hello world! How are you? I'm fine. This is a test. Another sentence!";
  std::vector<int> out(1, 0);

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создаем задачу
  auto sentenceCountTask = std::make_shared<fomin_v_sentence_count::SentenceCountSequential>(taskDataSeq);

  // Создаем атрибуты производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Количество запусков
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверяем результат
  ASSERT_EQ(5, out[0]);  // Ожидаемое количество предложений
}