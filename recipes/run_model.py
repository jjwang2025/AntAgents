#!/usr/bin/env python3
"""
大模型调用示例 - run_model.py

基于 OpenAI 兼容 API 的大模型调用实现，支持流式和非流式调用。
"""

import os
import sys
from typing import List, Optional, Generator, Dict, Any
from enum import Enum
import json
from dotenv import load_dotenv

from antagents import (
    OpenAIServerModel,
    ChatMessage,
    MessageRole,
    ChatMessageStreamDelta,
    TokenUsage
)


def create_example_messages() -> List[ChatMessage]:
    """创建示例消息"""
    return [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="你是一个有帮助的AI助手，请用中文回答用户的问题，回答要简洁明了。"
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="请解释一下人工智能的基本概念和应用领域。"
        )
    ]


def print_messages_detailed(messages: List[ChatMessage]):
    """美观地打印消息详情"""
    print("🔍 输入的Prompt详情:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for i, msg in enumerate(messages, 1):
        # 角色图标映射
        role_icons = {
            MessageRole.SYSTEM: "⚙️",
            MessageRole.USER: "👤", 
            MessageRole.ASSISTANT: "🤖",
            MessageRole.TOOL_CALL: "🛠️",
            MessageRole.TOOL_RESPONSE: "📋"
        }
        
        role_emoji = role_icons.get(msg.role, "📝")
        
        print(f"{role_emoji} 消息 {i} [{msg.role.value.upper()}]:")
        
        # 处理内容（支持多模态）
        if isinstance(msg.content, list):
            print(f"   📄 内容类型: 多模态 ({len(msg.content)} 个部分)")
            for j, part in enumerate(msg.content, 1):
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        content_preview = part.get('text', '')[:100] + '...' if len(part.get('text', '')) > 100 else part.get('text', '')
                        print(f"      {j}. 文本: {content_preview}")
                    elif part.get('type') in ['image', 'image_url']:
                        print(f"      {j}. 图像: [图像内容]")
        else:
            content_preview = str(msg.content)[:100] + '...' if msg.content and len(str(msg.content)) > 100 else msg.content
            print(f"   📄 内容: {content_preview}")
        
        # 显示工具调用
        if msg.tool_calls:
            print(f"   🛠️  工具调用: {len(msg.tool_calls)} 个")
            for tool_call in msg.tool_calls:
                if hasattr(tool_call, 'function'):
                    print(f"      - {tool_call.function.name}")
        
        # 显示token使用情况（如果有）
        if msg.token_usage:
            print(f"   📊 Token使用: 输入={msg.token_usage.input_tokens}, 输出={msg.token_usage.output_tokens}")
        
        if i < len(messages):
            print("   ────────────────────────────────────────")


def non_streaming_example(model: OpenAIServerModel, messages: List[ChatMessage]):
    """非流式调用示例"""
    print("\n🚀 开始非流式调用...")
    print("=" * 60)
    
    try:
        response = model.generate(messages=messages)
        
        print("✅ 非流式调用成功！")
        print("\n📋 模型回复:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        
        if response.token_usage:
            print("\n📊 Token 使用统计:")
            print(f"   📥 输入Token: {response.token_usage.input_tokens}")
            print(f"   📤 输出Token: {response.token_usage.output_tokens}")
            print(f"   📊 总Token: {response.token_usage.input_tokens + response.token_usage.output_tokens}")
        
        # 显示工具调用（如果有）
        if response.tool_calls:
            print("\n🛠️  工具调用:")
            for tool_call in response.tool_calls:
                if hasattr(tool_call, 'function'):
                    print(f"   - {tool_call.function.name}: {tool_call.function.arguments}")
            
    except Exception as e:
        print(f"❌ 非流式调用过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def streaming_example(model: OpenAIServerModel, messages: List[ChatMessage]):
    """流式调用示例"""
    print("\n🚀 开始流式调用...")
    print("=" * 60)
    
    try:
        print("📝 模型回复 (流式):")
        print("-" * 40)
        
        full_response = ""
        final_token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        tool_calls_accumulated = []
        
        for delta in model.generate_stream(messages=messages):
            if delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
            
            if delta.token_usage:
                final_token_usage.input_tokens += delta.token_usage.input_tokens
                final_token_usage.output_tokens += delta.token_usage.output_tokens
            
            # 处理工具调用（如果有）
            if delta.tool_calls:
                # 这里简化处理，实际使用时需要更复杂的累积逻辑
                tool_calls_accumulated.extend(delta.tool_calls)
        
        print("\n" + "-" * 40)
        
        print("\n📊 Token 使用统计:")
        print(f"   📥 输入Token: {final_token_usage.input_tokens}")
        print(f"   📤 输出Token: {final_token_usage.output_tokens}")
        print(f"   📊 总Token: {final_token_usage.input_tokens + final_token_usage.output_tokens}")
        
        # 显示工具调用（如果有）
        if tool_calls_accumulated:
            print("\n🛠️  工具调用检测到（流式模式下工具调用需要特殊处理）")
            
    except Exception as e:
        print(f"❌ 流式调用过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def interactive_chat(model: OpenAIServerModel):
    """交互式聊天示例"""
    print("\n💬 进入交互式聊天模式 (输入 'quit' 退出)")
    print("=" * 60)
    
    # 系统提示词
    system_message = ChatMessage(
        role=MessageRole.SYSTEM,
        content="你是一个有帮助的AI助手，请用中文进行友好、专业的对话。"
    )
    
    messages = [system_message]
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 退出交互模式")
                break
                
            if not user_input:
                continue
                
            # 添加用户消息
            user_message = ChatMessage(role=MessageRole.USER, content=user_input)
            messages.append(user_message)
            
            print("\n🤖 AI: ", end="", flush=True)
            
            # 流式响应
            response_content = ""
            token_usage = TokenUsage(input_tokens=0, output_tokens=0)
            
            for delta in model.generate_stream(messages=messages):
                if delta.content:
                    print(delta.content, end="", flush=True)
                    response_content += delta.content
                
                if delta.token_usage:
                    token_usage.input_tokens += delta.token_usage.input_tokens
                    token_usage.output_tokens += delta.token_usage.output_tokens
            
            # 添加AI回复到消息历史
            if response_content:
                assistant_message = ChatMessage(
                    role=MessageRole.ASSISTANT, 
                    content=response_content,
                    token_usage=token_usage
                )
                messages.append(assistant_message)
                
            # 显示本次调用的token使用
            print(f"\n   📊 本次调用Token: 输入={token_usage.input_tokens}, 输出={token_usage.output_tokens}")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出交互模式")
            break
        except Exception as e:
            print(f"\n❌ 聊天过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            break


def main():
    """主函数"""
    print("🤖 大模型调用示例程序")
    print("=" * 60)
    
    # 初始化模型
    try:
        model = OpenAIServerModel(
            model_id=os.getenv("DEEPSEEK_MODEL_ID"),
            api_base=os.getenv("DEEPSEEK_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        print(f"✅ 模型初始化成功: {model.model_id}")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建示例消息
    messages = create_example_messages()
    
    # 显示消息详情
    print_messages_detailed(messages)
    
    # 非流式调用示例
    success = non_streaming_example(model, messages)
    if not success:
        print("⚠️  非流式调用失败，跳过后续示例")
        return
    
    # 流式调用示例
    success = streaming_example(model, messages)
    if not success:
        print("⚠️  流式调用失败")
        return
    
    # 交互式聊天示例
    try:
        interactive_chat(model)
    except Exception as e:
        print(f"❌ 交互式聊天失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 程序执行完成！")


if __name__ == "__main__":
    load_dotenv()
    
    main()