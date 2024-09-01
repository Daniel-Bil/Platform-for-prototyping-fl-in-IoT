// SOME TUPLES NEED FIXING ex. Croping2D and 3D


export function generateStruct(name, s_id){
    const parametersCollector = {id: s_id, name:name}
    if ( name === "Activation" ) {
        parametersCollector.activation = {type:"string", values: ["relu", "sigmoid", "tanh", "XD"], default: "sigmoid"}

    } else if (name === "ActivityRegularization") {
        parametersCollector.l1 = { type: "float", values: [], default: 0.0 };
        parametersCollector.l2 = { type: "float", values: [], default: 0.0 };
    
    } else if (name === "Add") {
        // No specific parameters for Add layer
        
    } else if (name === "AdditiveAttention") {
        parametersCollector.use_scale = {type:"boolean", values: ["true", "false"], default: "true"}
        parametersCollector.dropout = {type:"float", values: [], default: 0.0}

    } else if (name === "AlphaDropout") {
        parametersCollector.rate = { type: "float", values: [], default: 0.1 };
        parametersCollector.noise_shape = { type: "tuple", values: [], default: null };
        parametersCollector.seed = { type: "int", values: [], default: null };
    
    } else if (name === "Attention") {
        parametersCollector.use_scale = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.causal = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.dropout = { type: "float", values: [], default: 0.0 };
    
    } else if (name === "Average") {
        // No specific parameters for Average layer
    
    } else if (name === "AveragePooling1D") {
        parametersCollector.pool_size = { type: "int", values: [], default: 2 };
        parametersCollector.strides = { type: "int", values: [], default: null };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    

    } else if (name === "AveragePooling2D") {
        parametersCollector.pool_size = { type: "tuple", values: [], default: [2, 2] };
        parametersCollector.strides = { type: "tuple", values: [], default: null };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "AveragePooling3D") {
        parametersCollector.pool_size = { type: "tuple", values: [], default: [2, 2, 2] };
        parametersCollector.strides = { type: "tuple", values: [], default: null };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };

    } else if (name === "BatchNormalization") {
        parametersCollector.axis = { type: "int", values: [], default: -1 };
        parametersCollector.momentum = { type: "float", values: [], default: 0.99 };
        parametersCollector.epsilon = { type: "float", values: [], default: 0.001 };
        parametersCollector.center = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.scale = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.beta_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.gamma_initializer = { type: "string", values: ["ones", "zeros", "random_normal", "random_uniform"], default: "ones" };
        parametersCollector.moving_mean_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.moving_variance_initializer = { type: "string", values: ["ones", "zeros", "random_normal", "random_uniform"], default: "ones" };
        parametersCollector.beta_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.gamma_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.beta_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.gamma_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

    } else if (name === "Bidirectional") {
        parametersCollector.layer = { type: "Layer", values: [], default: null };
        parametersCollector.merge_mode = { type: "string", values: [null, "sum", "mul", "concat", "ave"], default: "concat" };
        parametersCollector.backward_layer = { type: "Layer", values: [], default: null };
        parametersCollector.forward_layer = { type: "Layer", values: [], default: null };
    
    } else if (name === "CategoryEncoding") {
        parametersCollector.num_tokens = { type: "int", values: [], default: null };
        parametersCollector.output_mode = { type: "string", values: ["int", "binary", "count", "tf_idf"], default: "binary" };
        parametersCollector.sparse = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.count_weights = { type: "list", values: [], default: null };
    
    } else if (name === "CenterCrop") {
        parametersCollector.height = { type: "int", values: [], default: 0 };
        parametersCollector.width = { type: "int", values: [], default: 0 };
    
    } else if (name === "Concatenate") {
        parametersCollector.axis = { type: "int", values: [], default: -1 };

    } else if (name === "Conv1D") {
        parametersCollector.filters = { type: "int", values: [], default: 32 };
        parametersCollector.kernel_size = { type: "tuple", values: [], default: [3] };
        parametersCollector.strides = { type: "tuple", values: [], default: [1] };
        parametersCollector.padding = { type: "string", values: ["valid", "same", "causal"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
        parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1] };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

    } else if (name === "Conv1DTranspose") {
    parametersCollector.filters = { type: "int", values: [], default: 32 };
    parametersCollector.kernel_size = { type: "tuple", values: [], default: [3, 3] };
    parametersCollector.strides = { type: "tuple", values: [], default: [1, 1] };
    parametersCollector.padding = { type: "string", values: ["valid", "same", "causal"], default: "valid" };
    parametersCollector.output_padding = { type: "tuple", values: [], default: null };
    parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1] };
    parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
    parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
    parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
    parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
    parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
    parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

} else if (name === "Conv2D") {
    parametersCollector.filters = { type: "int", values: [], default: 32 };
    parametersCollector.kernel_size = { type: "tuple", values: [], default: [3, 3] };
    parametersCollector.strides = { type: "tuple", values: [], default: [1, 1] };
    parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
    parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1, 1] };
    parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
    parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
    parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
    parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
    parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
    parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

} else if (name === "Conv2DTranspose") {
    parametersCollector.filters = { type: "int", values: [], default: 32 };
    parametersCollector.kernel_size = { type: "tuple", values: [], default: [3, 3] };
    parametersCollector.strides = { type: "tuple", values: [], default: [1, 1] };
    parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
    parametersCollector.output_padding = { type: "tuple", values: [], default: null };
    parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1, 1] };
    parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
    parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
    parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
    parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
    parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
    parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

} else if (name === "Conv3D") {
    parametersCollector.filters = { type: "int", values: [], default: 32 };
    parametersCollector.kernel_size = { type: "tuple", values: [], default: [3, 3, 3] };
    parametersCollector.strides = { type: "tuple", values: [], default: [1, 1, 1] };
    parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
    parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1, 1, 1] };
    parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
    parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
    parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
    parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
    parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
    parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

} else if (name === "Conv3DTranspose") {
    parametersCollector.filters = { type: "int", values: [], default: 32 };
    parametersCollector.kernel_size = { type: "tuple", values: [], default: [3, 3, 3] };
    parametersCollector.strides = { type: "tuple", values: [], default: [1, 1, 1] };
    parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
    parametersCollector.output_padding = { type: "tuple", values: [], default: null };
    parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1, 1, 1] };
    parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
    parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
    parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
    parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
    parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
    parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
    parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

    } else if (name === "ConvLSTM1D") {

    } else if (name === "ConvLSTM2D") {

    } else if (name === "ConvLSTM3D") {

    } else if (name === "Cropping1D") {
        parametersCollector.cropping = { type: "tuple", values: [], default: [1, 1] };
    
    } else if (name === "Cropping2D") {
        parametersCollector.cropping = { type: "tuple2", values: [], default: [[0, 0], [0, 0]] };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "Cropping3D") {
        parametersCollector.cropping = { type: "tuple3", values: [], default: [[1, 1], [1, 1], [1, 1]] };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    

    } else if (name === "Dense") {
        parametersCollector.units = { type: "int", values: [], default: 128 };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
        parametersCollector.use_bias = { type: "boolean", values: [true, false], default: true };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

    } else if (name === "DepthwiseConv1D") {

    } else if (name === "DepthwiseConv2D") {
        parametersCollector.kernel_size = { type: "tuple", values: [], default: [3, 3] };
        parametersCollector.strides = { type: "tuple", values: [], default: [1, 1] };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.depth_multiplier = { type: "int", values: [], default: 1 };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
        parametersCollector.dilation_rate = { type: "tuple", values: [], default: [1, 1] };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.depthwise_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.depthwise_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.depthwise_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };    

    } else if (name === "Discretization") {

    } else if (name === "Dot") {

    } else if (name === "Dropout") {
        parametersCollector.rate = { type: "float", values: [], default: 0.5 };
        parametersCollector.noise_shape = { type: "tuple", values: [], default: null };
        parametersCollector.seed = { type: "int", values: [], default: null };
    
    } else if (name === "ELU") {
        parametersCollector.alpha = { type: "float", values: [], default: 1.0 };

    } else if (name === "EinsumDense") {

    } else if (name === "Embedding") {
        parametersCollector.input_dim = { type: "int", values: [], default: 1000 };
        parametersCollector.output_dim = { type: "int", values: [], default: 64 };
        parametersCollector.embeddings_initializer = { type: "string", values: ["uniform", "normal", "zeros", "ones", "glorot_uniform", "he_normal"], default: "uniform" };
        parametersCollector.embeddings_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.embeddings_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.mask_zero = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.input_length = { type: "int", values: [], default: null };

    } else if (name === "Flatten") {

    } else if (name === "FlaxLayer") {

    } else if (name === "GRU") {
        parametersCollector.units = { type: "int", values: [], default: 32 };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: "tanh" };
        parametersCollector.recurrent_activation = { type: "string", values: ["sigmoid", "hard_sigmoid"], default: "sigmoid" };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.recurrent_initializer = { type: "string", values: ["orthogonal", "glorot_uniform", "he_normal"], default: "orthogonal" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.recurrent_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.recurrent_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.recurrent_dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.return_sequences = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.return_state = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.go_backwards = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.stateful = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.unroll = { type: "boolean", values: ["true", "false"], default: "false" };
    
    } else if (name === "GRUCell") {
        parametersCollector.units = { type: "int", values: [], default: 32 };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: "tanh" };
        parametersCollector.recurrent_activation = { type: "string", values: ["sigmoid", "hard_sigmoid"], default: "sigmoid" };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.recurrent_initializer = { type: "string", values: ["orthogonal", "glorot_uniform", "he_normal"], default: "orthogonal" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.recurrent_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.recurrent_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.recurrent_dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.reset_after = { type: "boolean", values: ["true", "false"], default: "true" };

    } else if (name === "GaussianDropout") {
        parametersCollector.rate = { type: "float", values: [], default: 0.5 };
    
    } else if (name === "GaussianNoise") {
        parametersCollector.stddev = { type: "float", values: [], default: 0.1 };

    } else if (name === "GlobalAveragePooling1D") {

    } else if (name === "GlobalAveragePooling2D") {
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "GlobalAveragePooling3D") {
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };

    } else if (name === "GlobalMaxPooling1D") {
        // No specific parameters for GlobalMaxPooling1D
    
    } else if (name === "GlobalMaxPooling2D") {
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "GlobalMaxPooling3D") {
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };

    } else if (name === "GroupNormalization") {

    } else if (name === "GroupQueryAttention") {

    } else if (name === "HashedCrossing") {

    } else if (name === "Hashing") {

    } else if (name === "Identity") {

    } else if (name === "InputLayer") {
        parametersCollector.input_shape = { type: "tuple", values: [], default: null };
        parametersCollector.batch_size = { type: "int", values: [], default: null };
        parametersCollector.dtype = { type: "string", values: ["float32", "float64", "int32", "int64", "string", "bool"], default: "float32" };
        parametersCollector.sparse = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.ragged = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.name = { type: "string", values: [], default: null };

    } else if (name === "InputSpec") {

    } else if (name === "IntegerLookup") {

    } else if (name === "JaxLayer") {

    } else if (name === "LSTM") {

    } else if (name === "LSTMCell") {
        parametersCollector.units = { type: "int", values: [], default: 32 };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: "tanh" };
        parametersCollector.recurrent_activation = { type: "string", values: ["sigmoid", "hard_sigmoid"], default: "sigmoid" };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.recurrent_initializer = { type: "string", values: ["orthogonal", "glorot_uniform", "he_normal"], default: "orthogonal" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.unit_forget_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.recurrent_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.recurrent_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.recurrent_dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.implementation = { type: "int", values: [1, 2], default: 1 };

    } else if (name === "Lambda") {
        parametersCollector.function = { type: "callable", values: [], default: null };
        parametersCollector.output_shape = { type: "tuple", values: [], default: null };
        parametersCollector.mask = { type: "callable", values: [], default: null };
        parametersCollector.arguments = { type: "dict", values: [], default: {} };
    
    } else if (name === "Layer") {
        parametersCollector.trainable = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.name = { type: "string", values: [], default: null };
        parametersCollector.dtype = { type: "string", values: ["float32", "float64", "int32", "int64", "string", "bool"], default: null };

    } else if (name === "LayerNormalization") {

    } else if (name === "LeakyReLU") {
        parametersCollector.alpha = { type: "float", values: [], default: 0.3 };

    } else if (name === "Masking") {
        parametersCollector.mask_value = { type: "float", values: [], default: 0.0 };
    
    } else if (name === "MaxPooling1D") {
        parametersCollector.pool_size = { type: "int", values: [], default: 2 };
        parametersCollector.strides = { type: "int", values: [], default: null };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "MaxPooling2D") {
        parametersCollector.pool_size = { type: "tuple", values: [], default: "(2, 2)" };
        parametersCollector.strides = { type: "tuple", values: [], default: null };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };

    } else if (name === "MaxPooling3D") {
        parametersCollector.pool_size = { type: "tuple", values: [], default: [2,2,2] };
        parametersCollector.strides = { type: "tuple", values: [], default: null };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    

    } else if (name === "Maximum") {

    } else if (name === "MelSpectrogram") {

    } else if (name === "Minimum") {
        // Minimum layer does not have specific parameters.

    } else if (name === "MultiHeadAttention") {

    } else if (name === "Multiply") {
        // Multiply layer does not have specific parameters.

    } else if (name === "Normalization") {

    } else if (name === "PReLU") {
        parametersCollector.alpha_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.alpha_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.alpha_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.shared_axes = { type: "tuple", values: [], default: null };

    } else if (name === "Permute") {
        parametersCollector.dims = { type: "tuple", values: [], default: null };

    } else if (name === "RNN") {

    } else if (name === "RandomBrightness") {

    } else if (name === "RandomContrast") {

    } else if (name === "RandomCrop") {

    } else if (name === "RandomFlip") {

    } else if (name === "RandomHeight") {

    } else if (name === "RandomRotation") {

    } else if (name === "RandomTranslation") {

    } else if (name === "RandomWidth") {

    } else if (name === "RandomZoom") {

    } else if (name === "ReLU") {

    } else if (name === "RepeatVector") {

    } else if (name === "Rescaling") {

    } else if (name === "Reshape") { 
        parametersCollector.target_shape = { type: "tuple", values: [], default: [] };

    } else if (name === "Resizing") {

    } else if (name === "SeparableConv1D") {
        parametersCollector.filters = { type: "int", values: [], default: null };
        parametersCollector.kernel_size = { type: "int", values: [], default: null };
        parametersCollector.strides = { type: "int", values: [], default: 1 };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
        parametersCollector.dilation_rate = { type: "int", values: [], default: 1 };
        parametersCollector.depth_multiplier = { type: "int", values: [], default: 1 };
        parametersCollector.activation = { type: "string", values: [null, "relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.depthwise_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.pointwise_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.depthwise_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.pointwise_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.depthwise_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.pointwise_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
    
    } else if (name === "SeparableConv2D") {
        parametersCollector.filters = { type: "int", values: [], default: null };
        parametersCollector.kernel_size = { type: "tuple", values: [], default: null };
        parametersCollector.strides = { type: "tuple", values: [], default: "(1, 1)" };
        parametersCollector.padding = { type: "string", values: ["valid", "same"], default: "valid" };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
        parametersCollector.dilation_rate = { type: "tuple", values: [], default: "(1, 1)" };
        parametersCollector.depth_multiplier = { type: "int", values: [], default: 1 };
        parametersCollector.activation = { type: "string", values: [null, "relu", "sigmoid", "tanh", "softmax", "linear"], default: null };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.depthwise_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.pointwise_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.depthwise_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.pointwise_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.depthwise_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.pointwise_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };

    } else if (name === "SimpleRNN") {
        parametersCollector.units = { type: "int", values: [], default: 32 };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: "tanh" };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.recurrent_initializer = { type: "string", values: ["orthogonal", "glorot_uniform", "he_normal"], default: "orthogonal" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.recurrent_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.recurrent_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.recurrent_dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.return_sequences = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.return_state = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.go_backwards = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.stateful = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.unroll = { type: "boolean", values: ["true", "false"], default: "false" };
    
    } else if (name === "SimpleRNNCell") {
        parametersCollector.units = { type: "int", values: [], default: 32 };
        parametersCollector.activation = { type: "string", values: ["relu", "sigmoid", "tanh", "softmax", "linear"], default: "tanh" };
        parametersCollector.use_bias = { type: "boolean", values: ["true", "false"], default: "true" };
        parametersCollector.kernel_initializer = { type: "string", values: ["glorot_uniform", "he_normal", "lecun_normal", "random_normal", "random_uniform", "zeros", "ones"], default: "glorot_uniform" };
        parametersCollector.recurrent_initializer = { type: "string", values: ["orthogonal", "glorot_uniform", "he_normal"], default: "orthogonal" };
        parametersCollector.bias_initializer = { type: "string", values: ["zeros", "ones", "random_normal", "random_uniform"], default: "zeros" };
        parametersCollector.kernel_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.recurrent_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.bias_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.activity_regularizer = { type: "string", values: [null, "l1", "l2", "l1_l2"], default: null };
        parametersCollector.kernel_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.recurrent_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.bias_constraint = { type: "string", values: [null, "max_norm", "non_neg", "unit_norm"], default: null };
        parametersCollector.dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.recurrent_dropout = { type: "float", values: [], default: 0.0 };
        parametersCollector.implementation = { type: "int", values: [1, 2], default: 1 };
    
    } else if (name === "Softmax") {
        parametersCollector.axis = { type: "int", values: [], default: -1 };
    
    } else if (name === "SpatialDropout1D") {
        parametersCollector.rate = { type: "float", values: [], default: 0.5 };
    
    } else if (name === "SpatialDropout2D") {
        parametersCollector.rate = { type: "float", values: [], default: 0.5 };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "SpatialDropout3D") {
        parametersCollector.rate = { type: "float", values: [], default: 0.5 };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "SpectralNormalization") {
        parametersCollector.layer = { type: "layer", values: [], default: null };
        parametersCollector.power_iterations = { type: "int", values: [], default: 1 };
        parametersCollector.epsilon = { type: "float", values: [], default: 1e-12 };
    
    } else if (name === "StackedRNNCells") {
        parametersCollector.cells = { type: "list", values: [], default: [] };
    
    } else if (name === "StringLookup") {
        parametersCollector.max_tokens = { type: "int", values: [], default: null };
        parametersCollector.num_oov_indices = { type: "int", values: [], default: 1 };
        parametersCollector.mask_token = { type: "string", values: [], default: "[MASK]" };
        parametersCollector.oov_token = { type: "string", values: [], default: "[OOV]" };
        parametersCollector.invert = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.vocabulary = { type: "string", values: [], default: null };

    } else if (name === "Subtract") {
        // Subtract layer does not have specific parameters.
    
    } else if (name === "TFSMLayer") {
        parametersCollector.layer = { type: "layer", values: [], default: null };

    } else if (name === "TextVectorization") {
        parametersCollector.max_tokens = { type: "int", values: [], default: null };
        parametersCollector.standardize = { type: "string", values: ["lower_and_strip_punctuation", "none"], default: "lower_and_strip_punctuation" };
        parametersCollector.split = { type: "string", values: ["whitespace", "character"], default: "whitespace" };
        parametersCollector.ngram_range = { type: "list", values: [], default: [1, 1] };
        parametersCollector.output_mode = { type: "string", values: ["int", "binary", "count", "tf-idf"], default: "int" };
        parametersCollector.output_sequence_length = { type: "int", values: [], default: null };
        parametersCollector.pad_to_max_tokens = { type: "boolean", values: ["true", "false"], default: "false" };
        parametersCollector.vocabulary = { type: "string", values: [], default: null };

    } else if (name === "ThresholdedReLU") {
        parametersCollector.theta = { type: "float", values: [], default: 1.0 };
    
    } else if (name === "TimeDistributed") {
        parametersCollector.layer = { type: "Layer", values: [], default: null };

    } else if (name === "TorchModuleWrapper") {
        parametersCollector.module = { type: "Module", values: [], default: null };
        parametersCollector.trainable = { type: "boolean", values: ["true", "false"], default: "true" };
    
    } else if (name === "UnitNormalization") {
        parametersCollector.axis = { type: "int", values: [], default: -1 };
    
    } else if (name === "UpSampling1D") {
        parametersCollector.size = { type: "int", values: [], default: 2 };
    
    } else if (name === "UpSampling2D") {
        parametersCollector.size = { type: "list", values: [], default: [2, 2] };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
        parametersCollector.interpolation = { type: "string", values: ["nearest", "bilinear"], default: "nearest" };
    
    } else if (name === "UpSampling3D") {
        parametersCollector.size = { type: "list", values: [], default: [2, 2, 2] };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    

    } else if (name === "Wrapper") {
        parametersCollector.layer = { type: "Layer", values: [], default: null };
    
    } else if (name === "ZeroPadding1D") {
        parametersCollector.padding = { type: "int", values: [], default: 1 };
    
    } else if (name === "ZeroPadding2D") {
        parametersCollector.padding = { type: "list", values: [], default: [1, 1] };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };
    
    } else if (name === "ZeroPadding3D") {
        parametersCollector.padding = { type: "list", values: [], default: [1, 1, 1] };
        parametersCollector.data_format = { type: "string", values: [null, "channels_last", "channels_first"], default: null };

    } else {
        throw new Error("no Layer Name");
    }

    parametersCollector.kwargs = {type:"kwargs", values: [], default: null}
    return parametersCollector
}