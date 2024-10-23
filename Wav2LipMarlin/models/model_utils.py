def inspect_trainable_params(model):
    trainable_params = []
    non_trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            non_trainable_params.append((name, param.numel()))

    # Summarize trainable and non-trainable parameters
    total_trainable = sum([p[1] for p in trainable_params])
    total_non_trainable = sum([p[1] for p in non_trainable_params])

    print(f"Trainable Parameters: {total_trainable} params")
    print(f"Non-Trainable Parameters: {total_non_trainable} params")
    
    print("\nTrainable Parameters:")
    for name, num in trainable_params:
        print(f"{name}: {num} params")
    
    print("\nNon-Trainable Parameters:")
    for name, num in non_trainable_params:
        print(f"{name}: {num} params")

    # Example usage with your model
    #inspect_trainable_params(model)