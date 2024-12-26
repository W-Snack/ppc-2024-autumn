#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_sentence_count {

class SentenceCountSequential : public ppc::core::Task {
 public:
  explicit SentenceCountSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  char *input_;
  int sentence_count;
};

}  // namespace fomin_v_sentence_count