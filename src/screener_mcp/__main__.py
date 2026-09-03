"""
Main entrypoint for python -m screener_mcp.
If arguments are passed (e.g. python -m screener_mcp TCS), runs the fundamental analysis CLI.
If no arguments or --server flag, starts the MCP server.
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in {"--server", "server"}:
        from .cli import main as cli_main
        cli_main()
    else:
        from .server import main as server_main
        server_main()


if __name__ == "__main__":
    main()
