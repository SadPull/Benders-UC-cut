function o = llm_config()
%LLM_CONFIG 在这里填写一次，之后直接运行 result = run_suc();
o.llm_enabled = false; % 填好下面三项后改为 true
o.llm_url = '';        % 完整请求地址，例如 https://你的服务地址/v1/responses
o.llm_api_key = '';    % 填写你的 API Key
o.llm_model = '';      % 填写服务支持的模型名称

o.llm_timeout = 600;   % 单次请求超时（秒）
o.llm_patience = 3;    % 连续多少批次改善不足时调用 LLM
o.llm_max_calls = 3;   % 每次运行最多调用次数
end
