from A import data_preprocessing, Eff_model

if __name__ == "__main__":
    train_df, test_df=data_preprocessing.router()
    
    data_preprocessing.EDA(train_df)
    
    train_generator, valid_generator, target_size_dim = data_preprocessing.data_preprocess(train_df)
    
    model = Eff_model.create_model(target_size_dim)
    
    model.build(input_shape=(None, target_size_dim ,target_size_dim, 3))
    model.summary()
    
    reducelr, earlystop, model_check = Eff_model.callbacks()
    
    STEPS_PER_EPOCH = len(train_df)*0.8 / 64
    STEPS_PER_EPOCH = int(STEPS_PER_EPOCH)
    VALIDATION_STEPS = len(train_df)*0.2 / 64
    VALIDATION_STEPS = int(VALIDATION_STEPS)
    
    class_weights = Eff_model.class_weights(train_df)
    
    history = Eff_model.training(model, train_generator, valid_generator, STEPS_PER_EPOCH, VALIDATION_STEPS,reducelr, earlystop, model_check)
    
    Eff_model.acc_and_loss_plot(history)
    Eff_model.report_and_matrix(model, valid_generator)
    Eff_model.submission(test_df, model, target_size_dim)
