import (
	"context"
	"fmt"

	"google.golang.org/adk/agents"
	"google.golang.org/adk/model/llm/vllm"
)

func main() {
	client := &vllm.Client{
		BaseURL: "http://localhost:8001",
		Model:   "mistral",
		APIKey:  "dummy",
	}

	agent := agents.NewLlmAgentWithClient(client) // or whatever ctor your version exposes

	resp, err := agent.Run(context.Background(), "Explain vLLM in one sentence.")
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}
