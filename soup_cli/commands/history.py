"""soup history — DAG-style lineage view for a named artifact."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.tree import Tree

from soup_cli.registry.store import RegistryStore, validate_name

console = Console()


def history(
    name: str = typer.Argument(
        ..., help="Registry entry name to trace (e.g. 'medical-chat')",
    ),
) -> None:
    """Show the lineage tree for all entries with a given name."""
    try:
        validate_name(name)
    except ValueError as exc:
        console.print(f"[red]Invalid name: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    with RegistryStore() as store:
        entries = store.list_by_name(name)
        if not entries:
            console.print(
                f"[red]No registry entries named '{escape(name)}'.[/]"
            )
            raise typer.Exit(1)

        tree = Tree(f"[bold cyan]{escape(name)}[/]")
        for entry in entries:
            tag_str = ", ".join(entry.get("tags", [])) or "-"
            node = tree.add(
                f"[cyan]{escape(entry['id'])}[/] "
                f"([magenta]{escape(tag_str)}[/]) "
                f"[dim]{escape(entry['created_at'][:16])}[/]"
            )
            ancestors = store.get_ancestors(entry["id"], max_depth=5)
            if ancestors:
                anc_node = node.add("[dim]ancestors[/]")
                for anc in ancestors:
                    anc_node.add(
                        f"[cyan]{escape(anc['id'])}[/] "
                        f"[yellow]({escape(anc.get('relation', ''))})[/]"
                    )
            descendants = store.get_descendants(entry["id"], max_depth=5)
            if descendants:
                desc_node = node.add("[dim]descendants[/]")
                for desc in descendants:
                    desc_node.add(
                        f"[cyan]{escape(desc['id'])}[/] "
                        f"[yellow]({escape(desc.get('relation', ''))})[/]"
                    )

        console.print(tree)
