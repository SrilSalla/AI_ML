import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

import os

OPENAI_API_KEY = "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"

NOTION_API_KEY = "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"

SYSTEM_MESSAGE = "You are a helpful assistant that can search and summarize content from the user's Notion workspace and also list what is asked.Try to assume the tool and call the same and get the answer. Say TERMINATE when you are done with the task."

async def config():
    params = StdioServerParams(
        command="npx.cmd",
        args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'],
        env={
            'NOTION_API_KEY': NOTION_API_KEY
        },
        read_timeout_seconds=20
    )

    model = OpenAIChatCompletionClient(
        model="o4-mini",
        api_key=OPENAI_API_KEY,
    )

    mcp_tools = await mcp_server_tools(server_params=params)

    agent = AssistantAgent(
        name='notion_agent',
        system_message=SYSTEM_MESSAGE,
        model_client=model,
        tools=mcp_tools,
        reflect_on_tool_use=True
    )

    team = RoundRobinGroupChat(
        participants=[agent],
        max_turns=5,
        termination_condition=TextMentionTermination('TERMINATE')
    )

    return team

async def orchestrate(team,task):
    async for msg in team.run_stream(task=task):
        yield(msg)
        
async def main():
    team = await config()
    task = 'Create a new page titled "PageFromMCINotion" under parent page with ID "24962cf8359e8059892de4f4e902a6ad"'

    
    async for msg in orchestrate(team=team, task=task):
        print('-'*100)
        print(msg)
        print('-'*100)

if __name__ == "__main__":
    asyncio.run(main())