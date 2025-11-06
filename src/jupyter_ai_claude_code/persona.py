import os
from typing import Dict, List, AsyncIterator, Optional
import functools

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    ClaudeSDKClient,
)
from claude_code_sdk.types import McpHttpServerConfig


from jupyter_server.serverapp import ServerApp

from .templates import ClaudeCodeTemplateManager


AVATAR_PATH = os.path.join(os.path.dirname(__file__), "static", "claude.svg")


class ClaudeCodePersona(BasePersona):
    """Claude Code persona for Jupyter AI integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_mgr = ClaudeCodeTemplateManager(self)
        self._client: Optional[ClaudeSDKClient] = None

    async def connect(
        self, options: Optional[ClaudeCodeOptions] = None
    ) -> ClaudeSDKClient:
        """
        Initialize and connect a new ClaudeSDKClient for continuous conversation.

        Args:
            options: Optional ClaudeCodeOptions to configure the client

        Returns:
            The connected ClaudeSDKClient instance
        """
        if self._client is not None:
            self.log.warning("Client already connected")
            return self._client

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self.log.info("ClaudeSDKClient connected for continuous conversation")
        return self._client

    async def _get_or_create_client(
        self, options: ClaudeCodeOptions
    ) -> ClaudeSDKClient:
        """
        Get the existing client or create a new one with the given options.

        Args:
            options: ClaudeCodeOptions to configure the client

        Returns:
            The ClaudeSDKClient instance
        """
        if self._client is None:
            await self.connect(options)
        return self._client

    @property
    def defaults(self) -> PersonaDefaults:
        """Return default configuration for the Claude Code persona."""
        return PersonaDefaults(
            name="Claude",
            avatar_path=AVATAR_PATH,
            description="Claude Code persona",
            system_prompt="...",
        )

    async def _process_response_message(self, message_iterator) -> AsyncIterator[str]:
        """Process response messages with template updates."""
        has_content = False
        template_was_used = False

        async for message in message_iterator:
            self.log.info(str(message))
            if isinstance(message, AssistantMessage):
                result = await self.template_mgr.claude_message_to_str(message)
                # Template now handles everything - never stream individual components
                if self.template_mgr.active:
                    template_was_used = True
                elif result is not None:
                    # Only for messages without any tool usage (rare)
                    has_content = True
                    yield result + "\n\n"

        # Complete template if active
        if self.template_mgr.active:
            await self.template_mgr.complete()
            template_was_used = True

        # Always yield something to complete the stream
        if template_was_used:
            yield ""  # Empty yield to signal completion when template handled everything
        elif not has_content:
            yield ""  # Ensure stream completes for empty responses

    def _generate_prompt(self, message: Message) -> str:
        attachment_ids = message.attachments
        if attachment_ids is None:
            return message.body
        attachments = self.ychat.get_attachments()
        msg_attachments = (attachments[aid] for aid in attachment_ids)
        prompt = f"{message.body}\n\n"
        prompt += f"The user has attached the following files and may be referring to them in the above prompt:\n\n"
        for a in msg_attachments:
            if a["type"] == "file":
                prompt += f"file_path={a['value']}"
            elif a["type"] == "notebook":
                cells = list(c["id"] for c in a["cells"])
                # Claude Code's notebook tools only understand a single cell_id
                prompt += f"notebook_path={a['value']} cell_id={cells[0]}"
        self.log.info(prompt)
        return prompt

    @functools.cache
    def _get_mcp_servers_config(
        self,
    ) -> tuple[Dict[str, McpHttpServerConfig], List[str]]:
        """
        Auto-detect and configure MCP servers from Jupyter Server extensions.

        Checks if jupyter_server_mcp extension is available and adds it
        to the MCP server configuration along with allowed tools.

        Returns:
            Tuple of (mcp_servers_config, allowed_tools_list)
        """
        mcp_servers = {}
        allowed_tools = []
        # Check if jupyter_server_mcp extension is loaded
        try:
            server_app = ServerApp.instance()

            # Look for the MCP extension in the server app's extension manager
            if hasattr(server_app, "extension_manager"):
                extensions = server_app.extension_manager.extensions

                # Find jupyter_server_mcp extension
                mcp_extension = None
                for ext_name, ext_obj in extensions.items():
                    if (
                        ext_name == "jupyter_server_mcp"
                        or ext_obj.__class__.__name__ == "MCPExtensionApp"
                    ):
                        mcp_extension = ext_obj.extension_points[
                            "jupyter_server_mcp"
                        ].app
                        break

                if mcp_extension and hasattr(mcp_extension, "mcp_server_instance"):
                    # Extension is loaded and has an MCP server instance
                    mcp_server = mcp_extension.mcp_server_instance
                    if mcp_server:
                        # Configure MCP server connection
                        host = getattr(mcp_server, "host", "localhost")
                        port = getattr(mcp_server, "port", 3001)
                        name = getattr(mcp_server, "name", "Jupyter MCP Server")

                        server_config: McpHttpServerConfig = {
                            "type": "http",
                            "url": f"http://{host}:{port}/mcp",
                        }

                        mcp_servers[name] = server_config

                        # Get available tools from the MCP server
                        if (
                            hasattr(mcp_server, "_registered_tools")
                            and mcp_server._registered_tools
                        ):
                            # Add all tools from this server to allowed_tools
                            # Format: mcp__<serverName>__<toolName>
                            server_name_clean = name.replace(" ", "_").replace("-", "_")
                            for tool_name in mcp_server._registered_tools.keys():
                                allowed_tool = f"mcp__{server_name_clean}__{tool_name}"
                                allowed_tools.append(allowed_tool)

                            self.log.info(
                                f"Auto-configured MCP server: {name} at {server_config['url']} with {len(mcp_server._registered_tools)} tools"
                            )
                            self.log.debug(f"Allowed tools: {allowed_tools}")
                        else:
                            # If no specific tools, allow all tools from the server
                            server_name_clean = name.replace(" ", "_").replace("-", "_")
                            allowed_tools.append(f"mcp__{server_name_clean}")
                            self.log.info(
                                f"Auto-configured MCP server: {name} at {server_config['url']} (allowing all tools)"
                            )

        except Exception as e:
            self.log.error(f"Could not auto-detect MCP server: {e}")

        return mcp_servers, allowed_tools

    def _get_system_prompt(self):
        """Get the system prompt for Claude Code options."""
        return (
            "I am Claude Code, an AI assistant with access to development tools. "
            "When formatting responses, I use **bold text** for emphasis and section headers instead of markdown headings (# ## ###). "
            "I keep formatting clean and readable without large headers. "
            "For complex tasks requiring multiple steps (3+ actions), I proactively create a todo list using TodoWrite to track progress and keep the user informed of my plan."
        )

    async def process_message(self, message: Message) -> None:
        """Process incoming message and stream Claude Code response."""
        # Always set writing state at the start
        self.awareness.set_local_state_field("isWriting", True)

        self.template_mgr.reset()

        try:
            # Configure Claude Code - use workspace dir for better working directory detection
            chat_dir = self.get_chat_dir()
            workspace_dir = self.get_workspace_dir()

            # Prefer workspace dir if available, fallback to chat dir
            working_dir = chat_dir if chat_dir else workspace_dir

            self.log.info(f"Chat directory: {chat_dir}")
            self.log.info(f"Workspace directory: {workspace_dir}")
            self.log.info(f"Using working directory: {working_dir}")

            # Auto-detect and configure MCP servers and allowed tools
            mcp_servers, mcp_allowed_tools = self._get_mcp_servers_config()

            options = ClaudeCodeOptions(
                max_turns=50,
                cwd=working_dir,
                permission_mode="bypassPermissions",
                system_prompt=self._get_system_prompt(),
                mcp_servers=mcp_servers,
                allowed_tools=mcp_allowed_tools,
            )

            # Generate prompt from current message
            user_prompt = self._generate_prompt(message)

            # Get or create the client for continuous conversation
            client = await self._get_or_create_client(options)

            # Send the query and get response iterator
            await client.query(prompt=user_prompt)
            async_gen = client.receive_response()

            # Use stream_message to handle the streaming
            await self.stream_message(self._process_response_message(async_gen))

        except Exception as e:
            self.log.error(f"Error: {e}")
            if self.template_mgr.active:
                await self.template_mgr.complete()

            try:
                await self.send_message(f"Sorry, error: {e}")
            except TypeError:
                self.send_message(f"Sorry, error: {e}")
        finally:
            # Always clear writing state when done
            self.awareness.set_local_state_field("isWriting", False)
