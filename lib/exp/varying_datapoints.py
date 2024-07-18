



# TODO: Modify accordingly
def varying_datapoints(training_points = (10, 100, 1000, 10000)):

	epochs_max_classification = 10
	my_epoch_Classification = []
	my_train_cost_Classification = []
	my_val_cost_Classification = []
	my_test_cost_Classification = []

	for N_train_data in training_points:
		model = ClassificationConvNet(ConvNeuralNet(), ClassificationOutputLayer())

		sampled_train_loader, sampled_val_loader = get_random_sample_train_val(train_dataset, val_dataset, batch_size, N_train_data)
		optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

		# Update the train function call to get training costs
		my_epoch, my_train_cost, my_val_cost, my_test_cost = train(model, sampled_train_loader, sampled_val_loader, test_loader_original, cost_classification, optimizer, epochs_max_classification, acc_flag_classification, triplet_flag_classification, task_name_classification, N_train_data)

		my_epoch_Classification.append(my_epoch)
		my_train_cost_Classification.append(my_train_cost)  # Append the training costs
		my_val_cost_Classification.append(my_val_cost)  # Append the training costs
		my_test_cost_Classification.append(my_test_cost)  # Append the training costs



