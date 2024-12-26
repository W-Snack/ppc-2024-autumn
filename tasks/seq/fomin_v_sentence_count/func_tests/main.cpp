#include <gtest/gtest.h>

#include "seq/fomin_v_sentence_count/include/ops_seq.hpp"

TEST(Sequential, Test_Sentence_Count_Simple) {
  // Входная строка с 3 предложениями
  std::string input = "Hello world! How are you? I'm fine.";
  std::vector<int> out(1, 0);

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создаем задачу
  fomin_v_sentence_count::SentenceCountSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();

  // Проверяем результат
  ASSERT_EQ(3, out[0]);
}

TEST(Sequential, Test_Sentence_Count_Empty_String) {
  std::string input = "";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  fomin_v_sentence_count::SentenceCountSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();

  ASSERT_EQ(0, out[0]);
}

TEST(Sequential, Test_Sentence_Count_No_Sentences) {
  std::string input = "This is a string without any sentence delimiters";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  fomin_v_sentence_count::SentenceCountSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();

  ASSERT_EQ(0, out[0]);
}

TEST(Sequential, Test_Sentence_Count_Multiple_Delimiters) {
  // Строка с несколькими разделителями предложений подряд
  std::string input = "Hello!!! How are you?? I'm fine...";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  fomin_v_sentence_count::SentenceCountSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();

  ASSERT_EQ(3, out[0]);
}