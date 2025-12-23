package vllm

import (
	"google.golang.org/adk/model/llm"
	"google.golang.org/adk/model"
)

// RegisterModel registers a vLLM-backed LLM under the given name in the
// ADK model registry (if you're using that pattern).
// Adjust to match how you register models in your project.
func RegisterModel(name, baseURL, apiKey string) {
	model.RegisterLLM(name, func() llm.Client {
		return &Client{
			BaseURL: baseURL,
			Model:   name,
			APIKey:  apiKey,
		}
	})
}
