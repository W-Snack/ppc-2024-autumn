#include <gtest/gtest.h>

#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

TEST(fomin_v_sentence_count, Test_Empty_String) {
  boost::mpi::communicator world;
  std::string input = "";
  int result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  taskDataPar->outputs_count.emplace_back(1);

  // Create and run parallel task
  auto sentenceCountParallel = std::make_shared<fomin_v_sentence_count::SentenceCountParallel>(taskDataPar);
  ASSERT_EQ(sentenceCountParallel->validation(), true);
  sentenceCountParallel->pre_processing();
  sentenceCountParallel->run();
  sentenceCountParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(result, 0);
  }
}

TEST(fomin_v_sentence_count, Test_Single_Sentence) {
  boost::mpi::communicator world;
  std::string input = "Hello world.";
  int result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs_count.emplace_back(1);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
  }

  // Create and run parallel task
  auto sentenceCountParallel = std::make_shared<fomin_v_sentence_count::SentenceCountParallel>(taskDataPar);
  ASSERT_EQ(sentenceCountParallel->validation(), true);
  sentenceCountParallel->pre_processing();
  sentenceCountParallel->run();
  sentenceCountParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(result, 1);
  }
}

TEST(fomin_v_sentence_count, Test_Multiple_Sentences) {
  boost::mpi::communicator world;
  std::string input = "Hello world. How are you? I'm fine!";
  int result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs_count.emplace_back(1);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
  }

  // Create and run parallel task
  auto sentenceCountParallel = std::make_shared<fomin_v_sentence_count::SentenceCountParallel>(taskDataPar);
  ASSERT_EQ(sentenceCountParallel->validation(), true);
  sentenceCountParallel->pre_processing();
  sentenceCountParallel->run();
  sentenceCountParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(result, 3);
  }
}

TEST(fomin_v_sentence_count, Test_Sequential_Consistency) {
  boost::mpi::communicator world;
  std::string input = "This is a test. Another test! And one more?";
  int parallel_result = 0;
  int sequential_result = 0;

  // Create TaskData for parallel task
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs_count.emplace_back(1);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
  }

  // Create and run parallel task
  auto sentenceCountParallel = std::make_shared<fomin_v_sentence_count::SentenceCountParallel>(taskDataPar);
  ASSERT_EQ(sentenceCountParallel->validation(), true);
  sentenceCountParallel->pre_processing();
  sentenceCountParallel->run();
  sentenceCountParallel->post_processing();

  if (world.rank() == 0) {
    // Create TaskData for sequential task
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create and run sequential task
    fomin_v_sentence_count::SentenceCountSequential sentenceCountSequential(taskDataSeq);
    ASSERT_EQ(sentenceCountSequential.validation(), true);
    sentenceCountSequential.pre_processing();
    sentenceCountSequential.run();
    sentenceCountSequential.post_processing();

    // Compare results
    ASSERT_EQ(parallel_result, sequential_result);
  }
}
