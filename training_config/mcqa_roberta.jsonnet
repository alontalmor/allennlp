local train_size = 9741;
local batch_size = 8;
local gradient_accumulation_batch_size = 3;
local num_epochs = 4;
local learning_rate = 1e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-base";
local dataset_dir = "https://s3.amazonaws.com/commensenseqa/";
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "transformer_mc_qa",
    "sample": -1,
    "num_choices": 3,
    "context_syntax": "q#a!",
    "add_prefix": {"q": "Q: ", "a": "A: "},
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "validation_dataset_reader": {
    "type": "transformer_mc_qa",
    "sample": -1,
    "num_choices": 3,
    "context_syntax": "q#a!",
    "add_prefix": {"q": "Q: ", "a": "A: "},
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "datasets_for_vocab_creation": [],
  "train_data_path": dataset_dir + "train_rand_split.jsonl",
  "validation_data_path": dataset_dir + "dev_rand_split.jsonl",

  "model": {
    "type": "roberta_mc_qa",
    "pretrained_model": transformer_model
  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
      "type": "adam_w",
      "weight_decay": weight_decay,
      "betas": [0.9, 0.98],
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": warmup_ratio,
      "num_steps_per_epoch": std.ceil(train_size / gradient_accumulation_batch_size),
    },
    "validation_metric": "+EM",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    "gradient_accumulation_steps": gradient_accumulation_batch_size,
    // "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  }
}