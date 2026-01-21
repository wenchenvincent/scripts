## export AMD_LLM_API_KEY=<API KEY> before sourcing this script
curl -fsSL https://claude.ai/install.sh | bash -s latest

export PATH="$HOME/.local/bin:$PATH"

export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${AMD_LLM_API_KEY}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4.5"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4.1"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-3.5" # CC sends behind-the-scenes requests for things like, "make a title for this conversation." this makes these faster (optional)
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 # Disable telemetry and error reporting to anthropic

python3 -c "import json, os; p=os.path.expanduser('~/.claude.json'); d=json.load(open(p)) if os.path.exists(p) else {}; d['hasCompletedOnboarding']=True; json.dump(d, open(p,'w'), indent=2)"
