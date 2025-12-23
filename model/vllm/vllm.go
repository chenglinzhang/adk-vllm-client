// Package vllm implements the llm.Client interfaces for OpenAI-compatible
// vLLM models running locally or remotely.
package vllm

import (
	"context"
	"fmt"
	"io"

	openai "github.com/sashabaranov/go-openai"

	"google.golang.org/adk/model/llm"
)

// Client implements llm.Client, llm.ChatClient, and llm.StreamingChatClient
// for vLLM/OpenAI-compatible models.
type Client struct {
	// Base URL of the vLLM server, e.g. "http://localhost:8001".
	// This should NOT include "/v1" (it is added internally).
	BaseURL string

	// Model is the model identifier understood by vLLM, e.g. "mistral".
	Model string

	// APIKey is forwarded as Bearer token; vLLM commonly uses a dummy key.
	APIKey string

	// oa is lazily constructed; if non-nil, it is used directly.
	oa *openai.Client
}

var (
	_ llm.Client              = (*Client)(nil)
	_ llm.ChatClient          = (*Client)(nil)
	_ llm.StreamingChatClient = (*Client)(nil)
)

// newOpenAIClient constructs or returns the underlying OpenAI client
// configured to talk to the vLLM server.
func (c *Client) newOpenAIClient() (*openai.Client, error) {
	if c.oa != nil {
		return c.oa, nil
	}
	if c.BaseURL == "" {
		return nil, fmt.Errorf("vllm.Client.BaseURL is empty")
	}
	if c.Model == "" {
		return nil, fmt.Errorf("vllm.Client.Model is empty")
	}

	cfg := openai.DefaultConfig(c.APIKey)
	// vLLM exposes an OpenAI-compatible API at `${BaseURL}/v1`.
	cfg.BaseURL = c.BaseURL + "/v1"

	c.oa = openai.NewClientWithConfig(cfg)
	return c.oa, nil
}

// Chat implements llm.ChatClient.Chat using the OpenAI Chat Completions API.
// It sends the messages from llm.ChatRequest and returns the first choice
// as an llm.ChatResponse.
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
	oa, err := c.newOpenAIClient()
	if err != nil {
		return nil, err
	}

	messages := make([]openai.ChatCompletionMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    string(m.Role),
			Content: m.Content,
		})
	}

	resp, err := oa.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    c.Model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return nil, fmt.Errorf("vllm chat error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("vllm: no choices in response")
	}

	msg := resp.Choices[0].Message
	return &llm.ChatResponse{
		Message: llm.Message{
			Role:    llm.Role(msg.Role),
			Content: msg.Content,
		},
	}, nil
}

// ChatStream implements llm.StreamingChatClient.ChatStream using the
// OpenAI streaming Chat Completions API. It returns an llm.ChatStream
// that surfaces incremental deltas as ChatResponses.
func (c *Client) ChatStream(ctx context.Context, req *llm.ChatRequest) (llm.ChatStream, error) {
	oa, err := c.newOpenAIClient()
	if err != nil {
		return nil, err
	}

	messages := make([]openai.ChatCompletionMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    string(m.Role),
			Content: m.Content,
		})
	}

	stream, err := oa.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:    c.Model,
		Messages: messages,
		Stream:   true,
	})
	if err != nil {
		return nil, fmt.Errorf("vllm stream error: %w", err)
	}

	return &streamWrapper{stream: stream}, nil
}

// Name (optional) – if llm.Client in your version exposes Name(), you
// can add this for parity with Gemini-style clients.
// Remove this method if llm.Client does not define Name().
func (c *Client) Name() string {
	return c.Model
}

// streamWrapper adapts go-openai's ChatCompletionStream to llm.ChatStream.
type streamWrapper struct {
	stream *openai.ChatCompletionStream
}

var _ llm.ChatStream = (*streamWrapper)(nil)

// Recv reads the next delta from the streaming response. It returns:
//   - (*llm.ChatResponse, nil) on a non-empty content delta
//   - (nil, nil) on a non-content chunk (e.g., role-only)
//   - (nil, io.EOF) when the stream is done
//   - (nil, err) on error
func (s *streamWrapper) Recv() (*llm.ChatResponse, error) {
	chunk, err := s.stream.Recv()
	if err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, err
	}

	if len(chunk.Choices) == 0 {
		// No usable delta in this chunk; let caller loop again.
		return nil, nil
	}

	delta := chunk.Choices[0].Delta
	if delta.Content == "" {
		// Could just be a role or other metadata.
		return nil, nil
	}

	return &llm.ChatResponse{
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: delta.Content,
		},
	}, nil
}

// Close closes the underlying stream.
func (s *streamWrapper) Close() error {
	return s.stream.Close()
}
