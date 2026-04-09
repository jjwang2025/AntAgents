"""
计划定制示例

本示例演示了如何使用步骤回调在计划创建后中断智能体，
允许用户交互来批准或修改计划，然后恢复执行同时保留智能体记忆。

展示的关键概念：
1. 在PlanningStep后使用步骤回调中断
2. 提取和修改当前计划
3. 使用reset=False恢复执行以保留记忆
4. 用户交互进行计划批准/修改
"""

import os
from dotenv import load_dotenv

from antagents import OpenAIServerModel, ToolCallingAgent, WebSearchTool, PlanningStep


def display_plan(plan_content):
    """以格式化方式显示计划"""
    print("\n" + "=" * 60)
    print("🤖 AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice():
    """获取用户对计划批准的选择"""
    while True:
        choice = input("\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_modified_plan(original_plan):
    """允许用户修改计划"""
    print("\n" + "-" * 40)
    print("MODIFY PLAN")
    print("-" * 40)
    print("Current plan:")
    print(original_plan)
    print("-" * 40)
    print("Enter your modified plan (press Enter twice to finish):")

    lines = []
    empty_line_count = 0

    while empty_line_count < 2:
        line = input()
        if line.strip() == "":
            empty_line_count += 1
        else:
            empty_line_count = 0
        lines.append(line)

    # 移除最后两个空行
    modified_plan = "\n".join(lines[:-2])
    return modified_plan if modified_plan.strip() else original_plan


def interrupt_after_plan(memory_step, agent):
    """
    步骤回调函数，在计划步骤创建后中断智能体。
    这允许用户交互来审查并可能修改计划。
    """
    if isinstance(memory_step, PlanningStep):
        print("\n🛑 Agent interrupted after plan creation...")

        # 显示创建的计划
        display_plan(memory_step.plan)

        # 获取用户选择
        choice = get_user_choice()

        if choice == 1:  # 批准计划
            print("✅ Plan approved! Continuing execution...")
            # 不中断 - 让智能体继续执行
            return

        elif choice == 2:  # 修改计划
            # 从用户获取修改后的计划
            modified_plan = get_modified_plan(memory_step.plan)

            # 更新记忆步骤中的计划
            memory_step.plan = modified_plan

            print("\nPlan updated!")
            display_plan(modified_plan)
            print("✅ Continuing with modified plan...")
            # 不中断 - 让智能体继续执行修改后的计划
            return

        elif choice == 3:  # 取消
            print("❌ Execution cancelled by user.")
            agent.interrupt()
            return


def main():
    load_dotenv()
    
    """运行完整的计划定制示例"""
    print("🚀 Starting Plan Customization Example")
    print("=" * 60)

    # 创建启用了计划和步骤回调的智能体
    api_mode = os.getenv("OPENAI_API_MODE", "auto")
    model = OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_mode=api_mode,
    )
    tools = [WebSearchTool()]
    agent = ToolCallingAgent(model=model,
        tools=tools, # 添加搜索工具以获得更有趣的计划
        planning_interval=5,  # 每5步计划一次用于演示
        step_callbacks={PlanningStep: interrupt_after_plan},
        max_steps=10,
        provide_run_summary=True,
        verbosity_level=1)  # 显示智能体思考过程)
    
    # 定义一个能从计划中受益的任务
    task = """Search for recent developments in artificial intelligence and provide a summary
    of the top 3 most significant breakthroughs in 2025 which happen in China. Include the
    source of each breakthrough."""

    try:
        print(f"\n📋 Task: {task}")
        print("\n🤖 Agent starting execution...")

        # 第一次运行 - 将创建计划并可能被中断
        result = agent.run(task)

        # 如果执行到这里，说明计划已批准或执行已完成
        print("\n✅ Task completed successfully!")
        print("\n📄 Final Result:")
        print("-" * 40)
        print(result)

    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent execution was cancelled by user.")
            print("\nTo resume execution later, you could call:")
            print("agent.run(task, reset=False)  # This preserves the agent's memory")

            # 演示使用reset=False恢复执行
            print("\n" + "=" * 60)
            print("DEMONSTRATION: Resuming with reset=False")
            print("=" * 60)

            # 显示当前记忆状态
            print(f"\n📚 Current memory contains {len(agent.memory.steps)} steps:")
            for i, step in enumerate(agent.memory.steps):
                step_type = type(step).__name__
                print(f"  {i + 1}. {step_type}")

            # 询问用户是否想看恢复演示
            resume_choice = input("\nWould you like to see resume demonstration? (y/n): ").strip().lower()
            if resume_choice == "y":
                print("\n🔄 Resuming execution...")
                try:
                    # 恢复执行而不重置 - 保留记忆
                    agent.run(task, reset=False)
                    print("\n✅ Task completed after resume!")
                    print("\n📄 Final Result:")
                    print("-" * 40)
                except Exception as resume_error:
                    print(f"\n❌ Error during resume: {resume_error}")
                else:
                    print(f"\n❌ An error occurred: {e}")

    agent.replay(detailed=True)

if __name__ == "__main__":
    # 运行主示例
    main()
