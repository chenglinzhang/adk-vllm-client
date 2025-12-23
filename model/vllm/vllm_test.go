package vllm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/adk/model/llm"
)

func newTestClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	cfg := openai.DefaultConfig("test-key")
	cfg.BaseURL = srv.URL + "/v1"
	oa := openai.NewClientWithConfig(cfg)

	return &Client{
		BaseURL: srv.URL,
		Model:   "test-model",
		APIKey:  "test-key",
		oa:      oa, // inject our client
	}
}

func TestChat_Success(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: "hello from vllm",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	})

	ctx := context.Background()

	out, err := client.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "hi"},
		},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	if got, want := out.Message.Content, "hello from vllm"; got != want {
		t.Fatalf("unexpected content: got %q, want %q", got, want)
	}
	if got, want := out.Message.Role, llm.RoleAssistant; got != want {
		t.Fatalf("unexpected role: got %q, want %q", got, want)
	}
}

func TestChat_NoChoices(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{},
		}
		_ = json.NewEncoder(w).Encode(resp)
	})

	_, err := client.Chat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatalf("expected error for empty choices, got nil")
	}
}

func TestChatStream_Success(t *testing.T) {
	var callCount atomic.Int32

	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		w.Header().Set("Content-Type", "text/event-stream")

		// Simulate two chunks "hello " + "world"
		chunks := []openai.ChatCompletionStreamResponse{
			{
				Choices: []openai.ChatCompletionStreamChoice{
					{Delta: openai.ChatCompletionStreamChoiceDelta{Content: "hello "}},
				},
			},
			{
				Choices: []openai.ChatCompletionStreamChoice{
					{Delta: openai.ChatCompletionStreamChoiceDelta{Content: "world"}},
				},
			},
		}

		enc := json.NewEncoder(w)
		for _, c := range chunks {
			callCount.Add(1)
			_, _ = io.WriteString(w, "data: ")
			_ = enc.Encode(c)
			_, _ = io.WriteString(w, "\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.ChatStream(ctx, &llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "stream please"}},
	})
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	defer stream.Close()

	var result string
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv error: %v", err)
		}
		if resp == nil {
			continue
		}
		result += resp.Message.Content
	}

	if got, want := result, "hello world"; got != want {
		t.Fatalf("unexpected streamed content: got %q, want %q", got, want)
	}
	if got := callCount.Load(); got != 2 {
		t.Fatalf("expected 2 chunks, got %d", got)
	}
}

func TestChatStream_Error(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"something broke"}`))
	})

	stream, err := client.ChatStream(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		// go-openai returns errors synchronously on CreateChatCompletionStream
		t.Fatalf("expected error from ChatStream, got nil and stream=%#v", stream)
	}
}

