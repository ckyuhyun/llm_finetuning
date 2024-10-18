

class HyperParameterTuning:
    def __init__(self):
        pass


    def run_learn_rate_tune_per_layer(self):
        """
        Apply different learning rates to different layers of the model.
        This technique helps in adjusting the learning rates according to the specificity of each layer.
        :return:
        """
        # TODO : need to implement

        # sample snippet code
        # optimizer_grouped_parameters = [
        #     {'params': model.bert.encoder.layer[i].parameters(), 'lr': 1e-5 * (i + 1)}
        #     for i in range(len(model.bert.encoder.layer))
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        pass

    def run_unfreeze_layer_tune(self):
        """
        Gradually unfreeze the layers of the Transformer model during training.
        This approach helps in retaining the learned features in the early layers while fine-tuning the higher layers
        :return:
        """
        # TODO : need to implement
        # for layer in model.bert.encoder.layer:
        #     for param in layer.parameters():
        #         param.requires_grad = True
        #     trainer.train()
        pass

    def run_layer_wise_learning_rate_deca_tune(self):
        """
        Assign higher learning rates to the top layers and lower learning rates to the bottom layers.
        This technique ensures that the top layers adapt faster to the specific task
        :return:
        """
        # TODO : need to implement
        # optimizer_grouped_parameters = [
        #     {'params': model.bert.encoder.layer[i].parameters(), 'lr': 1e-5 / (i + 1)}
        #     for i in range(len(model.bert.encoder.layer))
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        pass

    def run_data_augmentation_tune(self):
        """
        Enhance your dataset by creating variations of the training data.
        This technique helps in making the model robust and improving generalization
        :return:
        """
        # TODO : need to implement
        # from nlpaug.augmenter.word import SynonymAug
        #
        # aug = SynonymAug(aug_src='wordnet')
        # augmented_text = aug.augment("The quick brown fox jumps over the lazy dog")
        pass

    def run_regularization_tune(self):
        """
        Use dropout and weight decay to prevent overfitting and improve model generalization
        :return:
        """
        # TODO : need to implement
        # training_args = TrainingArguments(
        #     output_dir='./results',
        #     num_train_epochs=3,
        #     per_device_train_batch_size=8,
        #     warmup_steps=500,  # Warm-up steps
        #     weight_decay=0.01,
        #     logging_dir='./logs',
        # )
        pass

    def run_warm_up_steps_tune(self):
        """
        Implement warm-up steps to gradually increase the learning rate at the beginning of training. This technique helps in stabilizing training.
        :return:
        """
        # TODO : need to implement
        # training_args = TrainingArguments(
        #     output_dir='./results',
        #     num_train_epochs=3,
        #     per_device_train_batch_size=8,
        #     warmup_steps=500,  # Warm-up steps
        #     weight_decay=0.01,
        #     logging_dir='./logs',
        # )
        pass


