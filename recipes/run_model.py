"""
云端大模型调用示例：

curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DeepSeek API Key>" \
  -d '{
        "model": "deepseek-chat",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "stream": false
      }'
"""
import os
from antagents.models import OpenAIServerModel, ChatMessage, MessageRole
from dotenv import load_dotenv

load_dotenv()

def print_messages_detailed(messages):
    """详细打印所有消息内容"""
    print("🔍 输入的Prompt详情:")
    print("━" * 60)
    
    for i, msg in enumerate(messages):
        if isinstance(msg, ChatMessage):
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            content = msg.content
            tool_calls = msg.tool_calls
        else:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            tool_calls = msg.get('tool_calls', None)
        
        # 使用不同的表情符号表示不同角色
        role_emoji = {
            'system': '⚙️',
            'user': '👤',
            'assistant': '🤖',
            'tool': '🛠️'
        }.get(role.lower(), '📝')
        
        print(f"{role_emoji} 消息 {i+1} [{role.upper()}]:")
        print(f"   📄 内容: {content}")
        
        if tool_calls:
            print(f"   🛠️  工具调用: {len(tool_calls)}个")
            for j, tool_call in enumerate(tool_calls):
                print(f"      {j+1}. {tool_call.function.name}({tool_call.function.arguments})")
        
        print("   " + "─" * 40)
    print()

def openai_model_example():
    """OpenAIServerModel 使用示例"""
    
    print("🚀 开始非流式调用示例")
    print("=" * 60)
    
    # 初始化OpenAI服务器模型
    model = OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,
        max_tokens=500
    )
    
    # 准备对话消息
    system_prompt = "你是一个有帮助的AI助手，请用中文回答用户的问题。回答要详细且有条理。"
    user_prompt = "请解释一下人工智能的基本概念和应用领域。"
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt)
    ]
    
    # 打印输入的prompt
    print_messages_detailed(messages)
    
    try:
        print("⏳ 正在调用模型...")
        # 调用模型生成响应
        response = model.generate(
            messages=messages,
            stop_sequences=["\n\n"],
        )
        
        # 输出结果
        print("✅ 模型响应结果:")
        print("━" * 60)
        print(f"🎭 角色: {response.role}")
        print(f"📋 内容:\n{response.content}")
        print(f"📊 Token使用: 输入{response.token_usage.input_tokens} / 输出{response.token_usage.output_tokens}")
        print(f"💾 总Token: {response.token_usage.input_tokens + response.token_usage.output_tokens}")
        
    except Exception as e:
        print(f"❌ 调用过程中出现错误: {e}")

def openai_stream_example():
    """流式生成示例"""
    
    print("\n🎬 开始流式调用示例")
    print("=" * 60)
    
    model = OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    
    system_prompt = "你是一个历史专家，请用生动有趣的方式介绍历史古迹。"
    user_prompt = "用100字介绍中国的长城"
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt)
    ]
    
    # 打印输入的prompt
    print_messages_detailed(messages)
    print("⏳ 流式响应 (实时输出):")
    print("━" * 60)
    
    try:
        full_response = ""
        for delta in model.generate_stream(messages=messages):
            if delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
            if delta.token_usage:
                print(f"\n\n📊 Token使用: 输入{delta.token_usage.input_tokens}/输出{delta.token_usage.output_tokens}")
                
        print(f"\n✅ 流式响应完成，总字数: {len(full_response)}")
        
    except Exception as e:
        print(f"\n❌ 流式调用错误: {e}")

if __name__ == "__main__":
    # 运行非流式示例
    openai_model_example()
    
    print("\n" + "="*60 + "\n")
    
    # 运行流式示例
    openai_stream_example()