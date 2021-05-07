from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2

class CustomTensorBoard(TensorBoard):
  def _log_weights(self, epoch):
    """
    This is the script for original _log_weights method of TensorBoard

    with self._train_writer.as_default():
      with summary_ops_v2.always_record_summaries():
        for layer in self.model.layers:
          for weight in layer.weights:
            weight_name = weight.name.replace(':', '_')
            summary_ops_v2.histogram(weight_name, weight, step=epoch)
            if self.write_images:
              self._log_weight_as_image(weight, weight_name, epoch)
        self._train_writer.flush()
    """
    with self._train_writer.as_default():
      with summary_ops_v2.always_record_summaries():
        for layer in self.model.layers:
          for weight in layer.weights:
            weight_name = 'My_Custom_Name_' + weight.name.replace(':', '_')
            summary_ops_v2.histogram(weight_name, weight, step=epoch)
            self._log_weight_as_image(weight, weight_name, epoch)
        self._train_writer.flush()


log_dir = "logs/custom_logs"
callback_func = CustomTensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_generator,
    epochs = 5,
    callbacks=[callback_func]
)


