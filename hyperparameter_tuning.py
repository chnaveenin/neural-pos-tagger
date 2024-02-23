from FFNN_runner import FFNNRunner
from LSTM_runner import LSTMRunner

class HyperparameterTuning:
    def __init__(self, train_file, val_file, test_file, flag=False, file=None):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.flag = flag
        self.embeddings = [100, 200]
        self.hidden_layers = [1, 2]
        self.hidden_layer_sizes = [50, 100]
        self.activation_functions = ['relu', 'tanh']
        self.context_window = [0, 1, 2, 3, 4]
        self.best_accuracy = 0
        self.best_config = None
        self.best_model = None
        self.file = file
        self.ps_accuracy = []
    
    def tuning(self):
        if self.flag:
            for emb in self.embeddings:
                for h_layer_size in self.hidden_layer_sizes:
                    for act_func in self.activation_functions:
                        trainer = FFNNRunner(
                            train_file=self.train_file,
                            val_file=self.val_file,
                            test_file=self.test_file,
                            embedding_dim=emb,
                            hidden_dim=h_layer_size,
                            num_epochs=2,
                            batch_size=1,
                            pre=2,
                            suc=2,
                            activation=act_func,
                            lr=0.001
                        )
                        accuracy = trainer.train_model()
                        f.write(f"Configuration: {emb, h_layer_size, act_func}, Accuracy: {accuracy}\n")
                        print(f"Configuration: {emb, h_layer_size, act_func}, Accuracy: {accuracy}")
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.best_config = (emb, h_layer_size, act_func)
                            self.best_model = trainer.model
            print(f"Best configuration-ffnn: {self.best_config}, Accuracy: {self.best_accuracy}")
            
            if self.best_config is not None:
                for _ in range(5):
                    _trainerforp = FFNNRunner(
                        train_file=self.train_file,
                        val_file=self.val_file,
                        test_file=self.test_file,
                        embedding_dim=self.best_config[0],
                        hidden_dim=self.best_config[1],
                        num_epochs=2,
                        batch_size=1,
                        pre=_,
                        suc=_,
                        activation=self.best_config[2],
                        lr=0.001
                    )
                    accuracy = _trainerforp.train_model()
                    f.write(f"Configuration: {self.best_config[0], self.best_config[1], _, _, self.best_config[2]}, Accuracy: {accuracy}\n")
                    print(f"Configuration: {self.best_config[0], self.best_config[1], _, _, self.best_config[2]}, Accuracy: {accuracy}")
                    self.ps_accuracy.append(accuracy)
            
        else:
            for emb in self.embeddings:
                for h_layer_size in self.hidden_layer_sizes:
                    for h_layers in self.hidden_layers:
                        for act_func in self.activation_functions:
                            trainer = LSTMRunner(
                                train_file=self.train_file,
                                val_file=self.val_file,
                                test_file=self.test_file,
                                embedding_dim=emb,
                                hidden_dim=h_layer_size,
                                num_epochs=2,
                                batch_size=1,
                                activation=act_func,
                                num_layers=h_layers,
                                lr=0.001
                            )
                            accuracy = trainer.train_model()
                            f.write(f"Configuration: {emb, h_layer_size, h_layers, act_func}, Accuracy: {accuracy}\n")
                            print(f"Configuration: {emb, h_layer_size, h_layers, act_func}, Accuracy: {accuracy}")
                            if accuracy > self.best_accuracy:
                                self.best_accuracy = accuracy
                                self.best_config = (emb, h_layer_size, h_layers, act_func)
                                        
with open('hyperparameter_tuning_results.txt', 'w') as f:
    f.write("FFNN Hyperparameter Tuning\n")
    print("FFNN Hyperparameter Tuning")
    h_tune = HyperparameterTuning("UD_English-Atis/en_atis-ud-train.conllu", "UD_English-Atis/en_atis-ud-dev.conllu", "UD_English-Atis/en_atis-ud-test.conllu", True, file=f)
    h_tune.tuning()

    f.write("\nLSTM Hyperparameter Tuning\n")
    print("LSTM Hyperparameter Tuning")
    h_tune = HyperparameterTuning("UD_English-Atis/en_atis-ud-train.conllu", "UD_English-Atis/en_atis-ud-dev.conllu", "UD_English-Atis/en_atis-ud-test.conllu", False, file=f)
    h_tune.tuning()