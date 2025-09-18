#include "embedding.h"
#include <iostream>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

const int NGL = 99;
#ifdef _WIN32
const std::string MODEL = "D:/lab-lsm-tree-handout/model/nomic-embed-text-v1.5.Q8_0.gguf";
#else
const std::string MODEL = "/d:/lab-lsm-tree-handout/model/nomic-embed-text-v1.5.Q8_0.gguf";
#endif
const int CONTEXT_SIZE = 2048;
const int BATCH_SIZE = 2048;
const int ROPE_SCALING_YARN = 1;
const float ROPE_FREQ_SCALE = 0.75;

// 添加全局变量，用于跟踪模型加载状态
static bool model_initialized = false;
static std::unique_ptr<llama_model_params> g_model_params;
static std::unique_ptr<llama_context_params> g_ctx_params;
static std::unique_ptr<llama_model, void(*)(llama_model*)> g_model(nullptr, [](llama_model* m) { if(m) llama_model_free(m); });
static std::unique_ptr<llama_context, void(*)(llama_context*)> g_ctx(nullptr, [](llama_context* c) { if(c) llama_free(c); });

// 初始化模型，只在第一次调用时执行
static bool initialize_model() {
    if (model_initialized) {
        return true;
    }
    
    common_params params;
    common_init();
    
    params.model = MODEL;
    params.n_gpu_layers = NGL;
    params.n_batch = BATCH_SIZE;
    params.n_ctx = CONTEXT_SIZE;
    params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    params.rope_freq_scale = ROPE_FREQ_SCALE;
    params.embedding = true;
    params.n_ubatch = params.n_batch;
    params.verbose_prompt = GGML_LOG_LEVEL_ERROR;
    
    llama_backend_init();
    llama_numa_init(params.numa);
    
    common_init_result llama_init = common_init_from_params(params);
    
    if (!llama_init.model) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return false;
    }
    
    // 保存模型和上下文
    g_model.reset(llama_init.model.release());
    g_ctx.reset(llama_init.context.release());
    
    model_initialized = true;
    std::cout << "Embedding model initialized successfully" << std::endl;
    return true;
}

// 在程序退出时释放资源
static void cleanup_model() {
    g_ctx.reset();
    g_model.reset();
    llama_backend_free();
    model_initialized = false;
}

// 实现清理函数，供外部调用
void embedding_cleanup() {
    cleanup_model();
}

std::string join(const std::vector<std::string>& vec,
                 const std::string& delimiter) {
  if (vec.empty()) return "";
  return std::accumulate(
      vec.begin() + 1, vec.end(), vec[0],
      [&delimiter](const std::string& a, const std::string& b) {
        return a + delimiter + b;
      });
}

static std::vector<std::string> split_lines(
    const std::string& s, const std::string& separator = "\n") {
  std::vector<std::string> lines;
  size_t start = 0;
  size_t end = s.find(separator);

  while (end != std::string::npos) {
    lines.push_back(s.substr(start, end - start));
    start = end + separator.length();
    end = s.find(separator, start);
  }

  lines.push_back(s.substr(start));  // Add the last part

  return lines;
}

static void batch_add_seq(llama_batch& batch,
                          const std::vector<int32_t>& tokens,
                          llama_seq_id seq_id) {
  size_t n_tokens = tokens.size();
  for (size_t i = 0; i < n_tokens; i++) {
    common_batch_add(batch, tokens[i], i, {seq_id}, true);
  }
}

static void batch_decode(llama_context* ctx, llama_batch& batch, float* output,
                         int n_seq, int n_embd, int embd_norm) {
  const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
  const struct llama_model* model = llama_get_model(ctx);

  // clear previous kv_cache values (irrelevant for embeddings)
  llama_kv_self_clear(ctx);

  // run model
  LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
  if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
    // encoder-only model
    if (llama_encode(ctx, batch) < 0) {
      LOG_ERR("%s : failed to encode\n", __func__);
    }
  } else if (!llama_model_has_encoder(model) &&
             llama_model_has_decoder(model)) {
    // decoder-only model
    if (llama_decode(ctx, batch) < 0) {
      LOG_ERR("%s : failed to decode\n", __func__);
    }
  }

  for (int i = 0; i < batch.n_tokens; i++) {
    if (!batch.logits[i]) {
      continue;
    }

    const float* embd = nullptr;
    int embd_pos = 0;

    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
      // try to get token embeddings
      embd = llama_get_embeddings_ith(ctx, i);
      embd_pos = i;
      GGML_ASSERT(embd != NULL && "failed to get token embeddings");
    } else {
      // try to get sequence embeddings - supported only when pooling_type is
      // not NONE
      embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
      embd_pos = batch.seq_id[i][0];
      GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
    }

    float* out = output + embd_pos * n_embd;
    common_embd_normalize(embd, out, n_embd, embd_norm);
  }
}

int embedding_utils(const std::string& prompt, std::vector<float>& embeddings,
                    int& n_embd, int& n_prompts) {
  // 暂停错误日志输出，减少控制台输出
  common_log_pause(common_log_main());
  
  // 初始化模型（如果尚未初始化）
  if (!initialize_model()) {
    return 1;
  }
  
  llama_model* model = g_model.get();
  llama_context* ctx = g_ctx.get();

  const llama_vocab* vocab = llama_model_get_vocab(model);

  const int n_ctx_train = llama_model_n_ctx_train(model);
  const int n_ctx = llama_n_ctx(ctx);

  const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

  if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
    LOG_ERR(
        "%s: computing embeddings in encoder-decoder models is not supported\n",
        __func__);
    return 1;
  }

  if (n_ctx > n_ctx_train) {
    LOG_WRN(
        "%s: warning: model was trained on only %d context tokens (%d "
        "specified)\n",
        __func__, n_ctx_train, n_ctx);
  }

  // 为了与原来代码兼容，创建一个本地参数对象
  common_params params;
  params.model = MODEL;
  params.n_gpu_layers = NGL;
  params.n_batch = BATCH_SIZE;
  params.n_ctx = CONTEXT_SIZE;
  params.embedding = true;

  // split the prompt into lines
  std::vector<std::string> prompts = split_lines(prompt, params.embd_sep);

  // max batch size
  const uint64_t n_batch = params.n_batch;
  GGML_ASSERT(params.n_batch >= params.n_ctx);

  // tokenize the prompts and trim
  std::vector<std::vector<int32_t>> inputs;
  for (const auto& prompt : prompts) {
    auto inp = common_tokenize(ctx, prompt, true, true);
    if (inp.size() > n_batch) {
      LOG_ERR(
          "%s: number of tokens in input line (%lld) exceeds batch size "
          "(%lld), increase batch size and re-run\n",
          __func__, (long long int)inp.size(), (long long int)n_batch);
      return 1;
    }
    inputs.push_back(inp);
  }

  // check if the last token is SEP
  // it should be automatically added by the tokenizer when
  // 'tokenizer.ggml.add_eos_token' is set to 'true'
  for (auto& inp : inputs) {
    if (inp.empty() || inp.back() != llama_vocab_sep(vocab)) {
      LOG_WRN("%s: last token in the prompt is not SEP\n", __func__);
      LOG_WRN(
          "%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the "
          "GGUF header\n",
          __func__);
    }
  }

  // initialize batch
  n_prompts = prompts.size();
  struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

  // count number of embeddings
  int n_embd_count = 0;
  if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
    for (int k = 0; k < n_prompts; k++) {
      n_embd_count += inputs[k].size();
    }
  } else {
    n_embd_count = n_prompts;
  }

  // allocate output
  n_embd = llama_model_n_embd(model);
  embeddings.resize(n_embd_count * n_embd, 0);
  float* emb = embeddings.data();

  // break into batches
  int e = 0;  // number of embeddings already stored
  int s = 0;  // number of prompts in current batch
  for (int k = 0; k < n_prompts; k++) {
    // clamp to n_batch tokens
    auto& inp = inputs[k];

    const uint64_t n_toks = inp.size();

    // encode if at capacity
    if (batch.n_tokens + n_toks > n_batch) {
      float* out = emb + e * n_embd;
      batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
      e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
      s = 0;
      common_batch_clear(batch);
    }

    // add to batch
    batch_add_seq(batch, inp, s);
    s += 1;
  }

  // final batch
  float* out = emb + e * n_embd;
  batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
  llama_perf_context_print(ctx);

  // clean up
  llama_batch_free(batch);

  return 0;
}

std::vector<std::vector<float>> embedding(const std::string& prompt) {
  int n_embd = 0;
  int n_prompts = 0;
  std::vector<float> embeddings;
  int ret = embedding_utils(prompt, embeddings, n_embd, n_prompts);
  if (ret != 0) {
    LOG_ERR("%s: failed to embed prompt\n", __func__);
    return std::vector<std::vector<float>>();
  }
  std::vector<std::vector<float>> out_embeddings;
  out_embeddings.resize(n_prompts, std::vector<float>(n_embd));
  for (int i = 0; i < n_prompts; i++) {
    out_embeddings[i] = std::vector<float>(
        embeddings.begin() + i * n_embd, embeddings.begin() + (i + 1) * n_embd);
  }
  return out_embeddings;
}

std::vector<float> embedding_single(const std::string& prompt) {
  // 如果字符串为空，返回空向量
  if (prompt.empty()) {
    return std::vector<float>();
  }
  
  // 移除对换行符的检查，改为使用embedding直接处理
  std::vector<std::vector<float>> embeddings = embedding(prompt);
  if (embeddings.empty()) {
    return std::vector<float>();
  }
  
  // 只返回第一个嵌入向量结果
  return embeddings[0];
}

std::vector<std::vector<float>> embedding_batch(const std::string& prompts) {
  // 确保每个文本都能被正确处理，即使含有换行符
  std::vector<std::string> text_list = split_lines(prompts);
  
  // 如果批处理为空，返回空结果
  if (text_list.empty()) {
    return std::vector<std::vector<float>>();
  }
  
  // 为每个文本生成嵌入
  std::vector<std::vector<float>> results;
  for (const auto& text : text_list) {
    // 跳过空字符串
    if (text.empty()) continue;
    
    auto single_embeddings = embedding_single(text);
    if (!single_embeddings.empty()) {
      results.push_back(single_embeddings);
    }
  }
  
  return results;
}
