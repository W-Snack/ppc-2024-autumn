#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

TEST(fomin_v_sentence_count, test_parallel_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_text;
  std::vector<int32_t> global_sentence_count(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_text = "Hello! How are you? I am fine.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_text.data()));
    taskDataPar->inputs_count.emplace_back(global_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sentence_count.data()));
    taskDataPar->outputs_count.emplace_back(global_sentence_count.size());
  }

  auto sentenceCountParallel = std::make_shared<fomin_v_sentence_count::SentenceCountParallel>(taskDataPar);
  ASSERT_EQ(sentenceCountParallel->validation(), true);
  sentenceCountParallel->pre_processing();
  sentenceCountParallel->run();
  sentenceCountParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(3, global_sentence_count[0]);  // Ожидаемое количество предложений
  }
}

TEST(fomin_v_sentence_count, test_sequential_task_run) {
  boost::mpi::communicator world;
  std::string global_text;
  std::vector<int32_t> global_sentence_count(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_text = "Hello! How are you? I am fine.";
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_text.data()));
    taskDataSeq->inputs_count.emplace_back(global_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sentence_count.data()));
    taskDataSeq->outputs_count.emplace_back(global_sentence_count.size());
  }

  auto sentenceCountSequential = std::make_shared<fomin_v_sentence_count::SentenceCountSequential>(taskDataSeq);
  ASSERT_EQ(sentenceCountSequential->validation(), true);
  sentenceCountSequential->pre_processing();
  sentenceCountSequential->run();
  sentenceCountSequential->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(3, global_sentence_count[0]);  // Ожидаемое количество предложений
  }
}
